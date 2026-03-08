"""Core API URL configuration."""

from django.urls import include, path
from rest_framework.authtoken.views import obtain_auth_token
from rest_framework.routers import DefaultRouter

from fuelsense.core.views import DashboardViewSet, DeliveryViewSet, FacilityViewSet, ModelRegistryViewSet, PlanningViewSet

router = DefaultRouter()
router.register(r"facilities", FacilityViewSet, basename="facility")
router.register(r"deliveries", DeliveryViewSet, basename="delivery")
router.register(r"models", ModelRegistryViewSet, basename="model-registry")

planning_trigger = PlanningViewSet.as_view({"post": "trigger"})
planning_history = PlanningViewSet.as_view({"get": "history"})
dashboard_kpis = DashboardViewSet.as_view({"get": "kpis"})
dashboard_drift = DashboardViewSet.as_view({"get": "drift_heatmap"})

urlpatterns = [
    path("", include(router.urls)),
    path("planning/trigger/", planning_trigger, name="planning-trigger"),
    path("planning/history/", planning_history, name="planning-history"),
    path("dashboard/kpis/", dashboard_kpis, name="dashboard-kpis"),
    path("dashboard/drift-heatmap/", dashboard_drift, name="dashboard-drift-heatmap"),
    path("auth/token/", obtain_auth_token, name="auth-token"),
]
