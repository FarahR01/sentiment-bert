import pandas as pd
import os
import logging
from pathlib import Path

# Paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
RAW_PATH = SCRIPT_DIR / "raw"
PROCESSED_PATH = SCRIPT_DIR / "processed"

def load_imdb_dataset(file_name: str) -> pd.DataFrame:
    """
    Load IMDB dataset from CSV/TSV
    """
    path = RAW_PATH / file_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    
    df = pd.read_csv(path)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing columns in dataset: {required_columns - set(df.columns)}")
    
    logging.info(f"Loaded {len(df)} rows from {file_name}")
    return df