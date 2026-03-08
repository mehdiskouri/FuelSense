"""FastAPI service entrypoint for route optimizer."""

from fastapi import FastAPI

app = FastAPI(title="FuelSense Optimizer", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "optimizer"}
