"""
Export BERT sentiment classifier to ONNX format.
"""

import torch
from app.models.bert_classifier import BertSentimentClassifier
import logging
import os


def export_to_onnx(output_path: str = "model.onnx"):
    """
    Exports the model to ONNX.
    """

    try:
        model_wrapper = BertSentimentClassifier()
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        # Dummy input for tracing
        dummy_text = "This is a sample sentence."
        inputs = tokenizer(
            dummy_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move to correct device
        device = torch.device(model_wrapper.settings.DEVICE)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)
        model.eval()

        # Export
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            opset_version=13,
            do_constant_folding=True
        )

        logging.info(f"ONNX model exported successfully to {output_path}")

    except Exception as e:
        logging.error(f"ONNX export failed: {str(e)}")
        raise RuntimeError("ONNX export failed.")


if __name__ == "__main__":
    os.makedirs("onnx_model", exist_ok=True)
    export_to_onnx("onnx_model/sentiment_bert.onnx")