import pandas as pd
import re
import logging

def clean_text(text: str) -> str:
    """
    Basic text cleaning for movie reviews.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Replace multiple spaces with single
    text = re.sub(r"\s+", " ", text)
    # Lowercase text (BERT uncased)
    text = text.lower()
    return text.strip()

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """
    Apply cleaning to the review column.
    """
    df[text_col] = df[text_col].apply(clean_text)
    logging.info("Text cleaning applied to dataframe.")
    return df