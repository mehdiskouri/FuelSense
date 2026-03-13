"""DRF views for FuelSense core app."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from celery import current_app
from django.core.cache import cache
from django.db.models import Avg
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from fuelsense.core.models import AnomalyAlert, Delivery, Facility, Forecast, InventoryLog, ModelRegistry, PlanningCycle
from fuelsense.core.reorder import filter_below_reorder
from fuelsense.core.serializers import (
    AnomalyAlertAcknowledgeSerializer,
    AnomalyAlertSerializer,
    DashboardKPISerializer,
    DeliveryDetailSerializer,
    DeliveryListSerializer,
    DeliveryStatusUpdateSerializer,
    DriftHeatmapSerializer,
    FacilityDetailSerializer,
    FacilityListSerializer,
    ForecastSerializer,
    InventoryLogSerializer,
    ModelRegistrySerializer,
    PlanningCycleSerializer,
    PlanningTriggerSerializer,
)

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from rest_framework.request import Request


logger = logging.getLogger(__name__)


class FacilityViewSet(viewsets.ModelViewSet[Facility]):
    """API endpoints for facility inventory, forecast, and alert data."""

    queryset = Facility.objects.select_related("fuel_type").all().order_by("id")
    permission_classes = (IsAuthenticated,)
    filterset_fields = ("fuel_type", "is_active", "facility_type")

    def get_serializer_class(self) -> type[FacilityListSerializer | FacilityDetailSerializer]:
        """Select list or detail serializer by action."""
        if self.action == "retrieve":
            return FacilityDetailSerializer
        return FacilityListSerializer

    @action(detail=True, methods=["get"], url_path="inventory")
    def inventory(self, request: Request, pk: str | None = None) -> Response:
        """Return recent inventory logs for one facility."""
        _ = pk
        facility = self.get_object()
        days = int(request.query_params.get("days", 90))
        cutoff = timezone.now() - timedelta(days=days)
        logs = facility.inventory_logs.filter(timestamp__gte=cutoff).order_by("-timestamp")
        return Response(InventoryLogSerializer(logs, many=True).data)

    @action(detail=True, methods=["get"], url_path="forecasts")
    def forecasts(self, _request: Request, pk: str | None = None) -> Response:
        """Return latest forecast for one facility."""
        _ = pk
        facility = self.get_object()
        latest = facility.forecasts.order_by("-created_at").first()
        if not latest:
            return Response({"detail": "No forecast available."}, status=status.HTTP_404_NOT_FOUND)
        return Response(ForecastSerializer(latest).data)

    @action(detail=True, methods=["get"], url_path="alerts")
    def alerts(self, request: Request, pk: str | None = None) -> Response:
        """Return anomaly alerts for one facility with optional filters."""
        _ = pk
        facility = self.get_object()
        qs = facility.alerts.all().order_by("-timestamp")
        anomaly_type = request.query_params.get("anomaly_type")
        ack = request.query_params.get("is_acknowledged")
        if anomaly_type:
            qs = qs.filter(anomaly_type=anomaly_type)
        if ack is not None:
            qs = qs.filter(is_acknowledged=ack.lower() == "true")
        paginator = PageNumberPagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        if page is not None:
            ser = AnomalyAlertSerializer(page, many=True)
            return paginator.get_paginated_response(ser.data)
        return Response(AnomalyAlertSerializer(qs, many=True).data)

    @action(detail=True, methods=["post"], url_path="acknowledge")
    def acknowledge(self, request: Request, pk: str | None = None) -> Response:
        """Acknowledge an alert linked to the selected facility."""
        _ = pk
        serializer = AnomalyAlertAcknowledgeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        facility = self.get_object()
        alert = facility.alerts.filter(id=serializer.validated_data["alert_id"]).first()
        if alert is None:
            return Response({"detail": "Alert not found for this facility."}, status=status.HTTP_404_NOT_FOUND)
        if not request.user.is_authenticated or request.user.pk is None:
            return Response({"detail": "Authentication required."}, status=status.HTTP_403_FORBIDDEN)
        alert.is_acknowledged = True
        alert.acknowledged_by_id = request.user.pk
        alert.notes = serializer.validated_data.get("notes", alert.notes)
        alert.save(update_fields=["is_acknowledged", "acknowledged_by", "notes"])
        return Response(AnomalyAlertSerializer(alert).data)


class DeliveryViewSet(viewsets.ModelViewSet[Delivery]):
    """API endpoints for delivery retrieval and status transitions."""

    queryset = (
        Delivery.objects.select_related("vehicle", "depot")
        .prefetch_related("items__facility__fuel_type")
        .order_by("-planned_date", "-id")
    )
    permission_classes = (IsAuthenticated,)
    filterset_fields = ("status", "vehicle", "depot")

    def get_serializer_class(self) -> type[DeliveryListSerializer | DeliveryDetailSerializer]:
        """Select list or detail serializer by action."""
        if self.action == "retrieve":
            return DeliveryDetailSerializer
        return DeliveryListSerializer

    @action(detail=True, methods=["post"], url_path="update-status")
    def update_status(self, request: Request, pk: str | None = None) -> Response:
        """Update delivery status while enforcing transition rules."""
        _ = pk
        delivery = self.get_object()
        serializer = DeliveryStatusUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        new_status = Delivery.Status(serializer.validated_data["status"])
        current_status = Delivery.Status(delivery.status)

        allowed = {
            Delivery.Status.PLANNED: {Delivery.Status.IN_TRANSIT, Delivery.Status.FAILED},
            Delivery.Status.IN_TRANSIT: {Delivery.Status.DELIVERED, Delivery.Status.FAILED},
            Delivery.Status.DELIVERED: set(),
            Delivery.Status.FAILED: set(),
        }
        if new_status not in allowed[current_status]:
            return Response({"detail": "Invalid status transition."}, status=status.HTTP_400_BAD_REQUEST)

        delivery.status = new_status
        delivery.save(update_fields=["status"])
        return Response(DeliveryDetailSerializer(delivery).data)


class PlanningViewSet(viewsets.ViewSet):
    """API endpoints to trigger and inspect planning cycles."""

    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=["post"], url_path="trigger")
    def trigger(self, request: Request) -> Response:
        """Queue manual or emergency planning execution."""
        serializer = PlanningTriggerSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        trigger_type = serializer.validated_data["trigger_type"]
        facility_ids = serializer.validated_data.get("facility_ids", [])

        if trigger_type == "EMERGENCY" and not facility_ids:
            return Response(
                {"facility_ids": ["Provide at least one facility id for EMERGENCY trigger."]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        cycle = PlanningCycle.objects.create(
            trigger_type=trigger_type,
            status=PlanningCycle.ExecutionStatus.QUEUED,
            facilities_in_queue=len(facility_ids) if trigger_type == "EMERGENCY" else 0,
            deliveries_created=0,
            total_distance_km=0.0,
            total_cost=0.0,
            solver_time_ms=0.0,
            baseline_cost=0.0,
            cost_reduction_pct=0.0,
        )

        if trigger_type == "EMERGENCY" and facility_ids:
            current_app.send_task(
                "fuelsense.core.tasks.run_emergency_planning_cycle",
                kwargs={"cycle_id": cycle.id, "facility_ids": facility_ids},
            )
        else:
            current_app.send_task("fuelsense.core.tasks.run_planning_cycle", kwargs={"cycle_id": cycle.id})

        return Response(
            {"planning_cycle_id": cycle.id, "status": cycle.status},
            status=status.HTTP_202_ACCEPTED,
        )

    @action(detail=False, methods=["get"], url_path="history")
    def history(self, _request: Request) -> Response:
        """Return paginated planning cycle history."""
        qs = PlanningCycle.objects.all().order_by("-triggered_at")
        page = PlanningViewSet.paginate_queryset(self, qs)
        if page is not None:
            ser = PlanningCycleSerializer(page, many=True)
            return self.get_paginated_response(ser.data)
        return Response(PlanningCycleSerializer(qs, many=True).data)

    def paginate_queryset(self, queryset: QuerySet[PlanningCycle]) -> list[PlanningCycle] | None:
        """Paginate queryset with local fallback paginator."""
        paginator = getattr(self, "paginator", None)
        if paginator is None:
            self.paginator = PageNumberPagination()
            self.paginator.page_size = 50
        return self.paginator.paginate_queryset(queryset, self.request, view=self)

    def get_paginated_response(self, data: object) -> Response:
        """Return a paginated API response for serialized records."""
        return self.paginator.get_paginated_response(data)


class ModelRegistryViewSet(viewsets.ModelViewSet[ModelRegistry]):
    """API endpoints for model registry operations."""

    queryset = ModelRegistry.objects.select_related("facility").all().order_by("-trained_at")
    serializer_class = ModelRegistrySerializer
    permission_classes = (IsAuthenticated,)
    filterset_fields = ("model_type", "is_active", "facility")

    @action(detail=True, methods=["post"], url_path="promote")
    def promote(self, _request: Request, pk: str | None = None) -> Response:
        """Promote selected model and deactivate its siblings."""
        _ = pk
        model = self.get_object()
        siblings = ModelRegistry.objects.filter(model_type=model.model_type, facility=model.facility).exclude(
            id=model.id,
        )
        siblings.update(is_active=False)
        model.is_active = True
        model.save(update_fields=["is_active"])
        return Response(ModelRegistrySerializer(model).data)

    @action(detail=True, methods=["post"], url_path="retrain")
    def retrain(self, _request: Request, pk: str | None = None) -> Response:
        """Enqueue retraining for the selected model scope."""
        _ = pk
        model = self.get_object()
        try:
            current_app.send_task(
                "fuelsense.core.tasks.retrain_model",
                args=[model.facility_id, model.model_type],
                ignore_result=True,
            )
        except Exception:
            logger.exception(
                "failed to enqueue retrain task",
                extra={"facility_id": model.facility_id, "model_type": model.model_type},
            )
        return Response({"status": "queued"}, status=status.HTTP_202_ACCEPTED)


class DashboardViewSet(viewsets.ViewSet):
    """API endpoints for dashboard KPI and drift visual data."""

    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=["get"], url_path="kpis")
    def kpis(self, _request: Request) -> Response:
        """Return cached dashboard KPI payload."""
        cache_key = "cache:dashboard:kpis"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached)

        thirty_days_ago = timezone.now() - timedelta(days=30)
        avg_delivery_cost = (
            Delivery.objects.filter(planned_date__gte=thirty_days_ago.date()).aggregate(v=Avg("total_cost"))["v"] or 0.0
        )
        avg_rmse = Forecast.objects.aggregate(v=Avg("rmse"))["v"] or 0.0
        total_alerts = AnomalyAlert.objects.count()
        unack = AnomalyAlert.objects.filter(is_acknowledged=False).count()
        below_reorder = filter_below_reorder(Facility.objects.all()).count()
        in_transit = Delivery.objects.filter(status=Delivery.Status.IN_TRANSIT).count()

        payload = {
            "avg_delivery_cost_last_30d": float(avg_delivery_cost),
            "forecast_accuracy_rmse": float(avg_rmse),
            "anomaly_detection_rate": float(total_alerts / max(InventoryLog.objects.count(), 1)),
            "unacknowledged_anomalies_count": int(unack),
            "facilities_below_reorder": int(below_reorder),
            "deliveries_in_transit": int(in_transit),
        }
        serializer = DashboardKPISerializer(data=payload)
        serializer.is_valid(raise_exception=True)
        cache.set(cache_key, serializer.data, timeout=300)
        return Response(serializer.data)

    @action(detail=False, methods=["get"], url_path="drift-heatmap")
    def drift_heatmap(self, _request: Request) -> Response:
        """Return drift ratios for latest registered models."""
        latest_models = ModelRegistry.objects.select_related("facility").all().order_by("-trained_at")
        payload = [
            {
                "facility_id": m.facility_id,
                "facility_name": m.facility.name if m.facility else None,
                "drift_ratio": m.drift_ratio,
                "model_version": m.version,
            }
            for m in latest_models
        ]
        serializer = DriftHeatmapSerializer(data=payload, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)
