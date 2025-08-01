from fastapi import APIRouter
import time
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()

@router.get("/latency")
async def measure_latency():
    """
    Measures single prediction latency.
    """
    test_text = "This is a dummy review for latency measurement."
    start = time.time()
    _ = service.classify_text(test_text)
    end = time.time()
    return {"latency_ms": (end-start)*1000}