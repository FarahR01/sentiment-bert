"""
BERT sentiment classification model wrapper.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.core.config import get_settings


class BertSentimentClassifier:
    """
    Encapsulates tokenizer + model inference logic.
    """

    def __init__(self):
        self.settings = get_settings()
        
        # Detect available device, fallback to CPU if CUDA not available
        if self.settings.DEVICE == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.settings.DEVICE)

        try:
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.MODEL_NAME
            )

            logging.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.MODEL_NAME,
                num_labels=2,
                ignore_mismatched_sizes=True 
            )

            self.model.to(self.device)
            self.model.eval()

            logging.info(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize BERT classifier.")

    def predict(self, text: str) -> dict:
        """
        Perform inference on a single text input.
        """

        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.settings.MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            return {
                "prediction": predicted_class,
                "confidence": probabilities[0][predicted_class].item()
            }

        except Exception as e:
            logging.error(f"Inference failed: {str(e)}")
            raise RuntimeError("Prediction failed.")