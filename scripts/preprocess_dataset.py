import pandas as pd
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data.clean import preprocess_dataframe
from app.data.split import stratified_split

RAW_PATH = "app/data/raw/IMDB.csv"
PROCESSED_PATH = "app/data/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)

df = pd.read_csv(RAW_PATH)
df = preprocess_dataframe(df)

train_df, val_df, test_df = stratified_split(df)

train_df.to_csv(os.path.join(PROCESSED_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_PATH, "val.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_PATH, "test.csv"), index=False)

print("Preprocessing complete. Files saved to app/data/processed/")
print(train_df.head()["review"])
print(train_df["sentiment"].value_counts(normalize=True))