"""PharmaIQ FastAPI application entry point.

Run from the repository root:  uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="PharmaIQ API",
    version="0.1.0",
    description="Synthetic pharmaceutical sales analytics, forecasting, segmentation and "
                "scenario-based budget optimization. All data is synthetic (synthetic_v1).",
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness: the process is up and serving requests (no dependency checks)."""
    return {"status": "ok"}
