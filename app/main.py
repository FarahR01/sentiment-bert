"""
Main entry point for FastAPI application.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import logging
import torch

from app.api.schemas import PredictionRequest, PredictionResponse
from app.core.config import get_settings
from app.core.logger import configure_logging
from app.utils.reproducibility import set_global_seed
from app.services.prediction_service import PredictionService
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.api.monitor import router as monitor_router

class TextInput(BaseModel):
    """Request model for prediction endpoint."""
    text: str


def create_app() -> FastAPI:
    """
    Application factory pattern.
    """

    settings = get_settings()
    prediction_service = PredictionService()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.API_VERSION
    )

    @app.on_event("startup")
    async def startup_event():
        """
        Startup lifecycle hook.
        """

        configure_logging()
        set_global_seed(42)

        # Validate device
        if settings.DEVICE == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA selected but not available. Switching to CPU.")
            settings.DEVICE = "cpu"

        logging.info(f"Application started on device: {settings.DEVICE}")

    @app.get("/health")
    async def health_check():
        """
        Health check endpoint.
        """
        return {"status": "ok"}
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Sentiment prediction endpoint.
        """

        try:
            result = await run_in_threadpool(
                prediction_service.classify_text,
                request.text
            )

            return result

        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    # @app.post("/predict")
    # async def predict(input_data: TextInput):
    #     """
    #     Sentiment prediction endpoint.
    #     Expects a JSON payload with a "text" field.
    #     """
    #     try:
    #         result = prediction_service.classify_text(input_data.text)
    #         return result
    #     except Exception as e:
    #         logging.error(f"Prediction error: {str(e)}")
    #         return {"error": "Prediction failed"}
    return app


app = create_app()

app.include_router(monitor_router)


if __name__ == "__main__":
    from app.data.ingest import load_imdb_dataset
    
    df = load_imdb_dataset("IMDB.csv")
    print(df.head())
    print(df['sentiment'].value_counts())