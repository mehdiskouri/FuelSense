"""FastAPI service entrypoint for demand forecaster."""

from fastapi import FastAPI

app = FastAPI(title="FuelSense Forecaster", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "forecaster"}
