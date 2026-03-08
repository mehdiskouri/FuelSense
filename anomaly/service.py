"""FastAPI service entrypoint for anomaly detector."""

from fastapi import FastAPI

app = FastAPI(title="FuelSense Anomaly", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "anomaly"}
