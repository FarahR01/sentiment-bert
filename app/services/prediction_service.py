"""
Service layer responsible for business logic.
"""

import logging
from app.models.bert_classifier import BertSentimentClassifier


class PredictionService:
    """
    Handles prediction workflow.
    """

    def __init__(self):
        try:
            self.model = BertSentimentClassifier()
        except Exception as e:
            logging.error("PredictionService initialization failed.")
            raise

    def classify_text(self, text: str) -> dict:
        """
        Classifies input text.
        """

        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        return self.model.predict(text)