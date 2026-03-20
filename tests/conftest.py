"""
Pytest configuration and shared fixtures for integration tests.
Provides sample data and common setup for ML pipeline tests.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_imdb_data():
    """Create representative sample IMDB dataset for testing."""
    data = {
        "review": [
            "This movie was absolutely excellent. Highly recommend!",
            "Terrible film. Complete waste of time.",
            "Amazing cinematography and great acting throughout.",
            "Boring and predictable. Very disappointed.",
            "Outstanding performance by the lead actor.",
            "Poor plot development and weak dialogue.",
            "One of the best movies I've ever seen!",
            "Disappointing and overrated.",
            "Fantastic story with perfect execution.",
            "Awful movie. Don't bother watching.",
            "Brilliant direction and compelling narrative.",
            "Unwatchable. Worst film ever made.",
            "Superb entertainment from start to finish.",
            "Dull and forgettable experience.",
            "Masterpiece. A true gem of cinema.",
        ],
        "sentiment": [
            "positive", "negative", "positive", "negative", "positive",
            "negative", "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative", "positive"
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_cleaned_data(sample_imdb_data):
    """Apply cleaning transformations to sample data."""
    import re
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        return text.strip()
    
    df = sample_imdb_data.copy()
    df["review"] = df["review"].apply(clean_text)
    return df


@pytest.fixture
def sample_train_test_split(sample_cleaned_data):
    """Create train/test split from sample data."""
    from sklearn.model_selection import train_test_split
    
    df = sample_cleaned_data
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["sentiment"], random_state=42
    )
    return train_df, test_df


@pytest.fixture
def test_data_path(temp_data_dir, sample_imdb_data):
    """Create test data files in temporary directory."""
    data_dir = Path(temp_data_dir) / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_df, temp = train_test_split(
        sample_imdb_data, test_size=0.4, stratify=sample_imdb_data["sentiment"], 
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp, test_size=0.5, stratify=temp["sentiment"], random_state=42
    )
    
    # Save CSVs
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "root": str(data_dir)
    }


@pytest.fixture
def mock_model(tmp_path):
    """Create a mock BERT model for testing without loading actual pretrained weights."""
    import torch
    from transformers import AutoTokenizer
    
    class MockBERTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(30522, 768)
            self.classifier = torch.nn.Linear(768, 2)
        
        def forward(self, input_ids, attention_mask=None, labels=None):
            # Simple mock: average embeddings and classify
            embeddings = self.embedding(input_ids)
            pooled = embeddings.mean(dim=1)
            logits = self.classifier(pooled)
            
            class Output:
                def __init__(self, logits, loss=None):
                    self.logits = logits
                    self.loss = loss
            
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
                return Output(logits, loss)
            return Output(logits)
    
    return MockBERTModel()
