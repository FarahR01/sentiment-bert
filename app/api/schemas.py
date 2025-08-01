"""
Pydantic request and response schemas.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for sentiment prediction.
    """
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Movie review text"
    )


class PredictionResponse(BaseModel):
    """
    Output schema for sentiment prediction.
    """
    prediction: int
    confidence: float