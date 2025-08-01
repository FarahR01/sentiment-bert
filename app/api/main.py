from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="BERT Sentiment ONNX API")

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
session = ort.InferenceSession("app/models/bert_sentiment.onnx")

# Request schema
class ReviewRequest(BaseModel):
    text: str

# Response schema
class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Run ONNX inference
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    logits = outputs[0][0]
    pred_idx = int(np.argmax(logits))
    confidence = float(np.max(logits))  # Optional: raw logit score

    sentiment = "positive" if pred_idx == 1 else "negative"

    return ReviewResponse(sentiment=sentiment, confidence=confidence)