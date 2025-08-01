import torch
import onnx
import os
import sys
from transformers import BertForSequenceClassification, AutoTokenizer

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

MODEL_PATH = "app/models/best_model.pt"
ONNX_PATH = "app/models/bert_sentiment.onnx"

device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dummy_text = "This movie was amazing!"

inputs = tokenizer(
    dummy_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"}
    },
    opset_version=18,
    dynamo=False,  # avoids dynamic_axes+dynamo warning
)

m = onnx.load(ONNX_PATH)
print(f"ONNX model exported successfully! opset={m.opset_import[0].version}")