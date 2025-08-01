import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class IMDBDataset(Dataset):

    def __init__(self, csv_path, max_length=256):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = self.data.iloc[idx]["review"]
        sentiment = self.data.iloc[idx]["sentiment"]
        label = 1 if sentiment == "positive" else 0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }