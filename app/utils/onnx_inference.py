"""
ONNX Runtime inference for latency testing.
"""

import onnxruntime as ort
from transformers import AutoTokenizer
import torch
import time
import logging

MODEL_PATH = "onnx_model/sentiment_bert.onnx"
DEVICE = "cpu"  # or "cuda" if GPU

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"] if DEVICE=="cpu" else ["CUDAExecutionProvider"])

def predict_onnx(text: str):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    ort_inputs = {k: v for k, v in inputs.items()}

    start = time.time()
    outputs = session.run(None, ort_inputs)
    end = time.time()

    logits = outputs[0]
    predicted_class = int(logits.argmax(axis=1)[0])
    latency_ms = (end - start) * 1000

    return predicted_class, latency_ms


if __name__ == "__main__":
    test_text = "Oh great, another boring movie..."
    pred, latency = predict_onnx(test_text)
    logging.info(f"Prediction: {pred}, Latency: {latency:.2f} ms")