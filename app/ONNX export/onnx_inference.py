import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

session = ort.InferenceSession("app/models/bert_sentiment.onnx")

text = "This movie was fantastic!"

inputs = tokenizer(
    text,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=128
)

outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
)

logits = outputs[0]
prediction = np.argmax(logits)

print("Prediction:", prediction)