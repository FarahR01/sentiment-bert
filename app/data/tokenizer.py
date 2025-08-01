from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(texts, max_length=256):
    """
    Tokenize a list of texts into input_ids and attention_mask.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
