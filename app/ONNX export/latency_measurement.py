import time
import os
import sys
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
session = ort.InferenceSession("app/models/bert_sentiment.onnx")

# Create sample inputs
sample_text = "This movie was absolutely amazing!"
inputs = tokenizer(
    sample_text,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=128
)

# Warm up
session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
})

# Measure latency
start = time.time()

for _ in range(100):
    session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

end = time.time()

print(f"Average latency: {(end-start)/100 * 1000:.2f} ms")