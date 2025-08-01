# Data Pipeline

## Overview

The data pipeline handles all stages of data processing from raw IMDB dataset to cleaned, tokenized datasets ready for model training and evaluation.

## Pipeline Stages

### 1. Data Ingestion (`app/data/ingest.py`)

**Purpose**: Load raw IMDB dataset into memory

**Process**:
```
Raw IMDB Dataset (CSV)
        ↓
Load into pandas DataFrame
        ↓
Validation & basic checks
```

**Key Functions**:
- Load dataset from CSV file
- Validate structure (columns, data types)
- Handle missing values
- Compute dataset statistics

**Input**: `app/data/raw/IMDB.csv`
**Output**: DataFrame with columns: `text`, `label`

### 2. Data Cleaning (`app/data/clean.py`)

**Purpose**: Clean and normalize text data

**Transformations Applied**:
- Remove HTML tags and special characters
- Convert to lowercase
- Remove extra whitespace
- Remove URLs
- Normalize punctuation
- Handle contractions (e.g., "don't" → "do not")
- Remove stop words (optional, configurable)

**Quality Checks**:
- Minimum review length (characters)
- Maximum review length (for truncation)
- Check for empty reviews after cleaning
- Verify label distribution

**Output**: Cleaned DataFrame ready for splitting

### 3. Data Splitting (`app/data/split.py`)

**Purpose**: Partition data into train, validation, and test sets

**Strategy**: Stratified split to maintain class balance

**Distribution**:
- **Train**: 70% of data
- **Validation**: 15% of data
- **Test**: 15% of data

**Process**:
```
Cleaned Dataset
    ↓
Stratified split (maintains label ratio)
    ↓
Train Set → train.csv
Val Set   → val.csv
Test Set  → test.csv
```

**Output Location**: `app/data/processed/`

### 4. Tokenization (`app/data/tokenizer.py`)

**Purpose**: Convert text to BERT tokens and create training datasets

**Steps**:
1. Load BERT tokenizer from HuggingFace
2. Tokenize all text reviews
3. Pad sequences to fixed length (typically 512 tokens for BERT)
4. Create attention masks
5. Prepare PyTorch datasets with labels

**Tokenizer**: `bert-base-uncased`

**Features**:
- Special tokens: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`
- Maximum sequence length: 512 tokens
- Padding strategy: Right padding
- Truncation: Enabled for overly long sequences

**Output**: PyTorch Dataset objects ready for DataLoader

## Data Flow Diagram

```
IMDB.csv (raw)
    ↓
[Data Ingestion]
    ↓
[Data Cleaning] → Remove noise, normalize
    ↓
[Data Splitting] → Stratified split
    ├── train.csv (70%)
    ├── val.csv (15%)
    └── test.csv (15%)
    ↓
[Tokenization] → BERT tokens + attention masks
    ↓
[PyTorch Datasets]
    ↓
[DataLoaders] → Ready for training
```

## Dataset Statistics

After processing, the dataset should have:

| Metric | Value |
|--------|-------|
| Total Samples | ~25,000 |
| Positive Reviews | ~12,500 (50%) |
| Negative Reviews | ~12,500 (50%) |
| Avg. Review Length | 200-300 tokens |
| Max Sequence Length | 512 tokens |

## Configuration

Key parameters defined in `app/core/config.py`:

```python
# Text cleaning
MIN_REVIEW_LENGTH = 10
MAX_REVIEW_LENGTH = 512

# Data splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Tokenization
MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 32
```

## Running the Pipeline

```bash
# Ingest raw data
python -c "from app.data.ingest import load_imdb_data; load_imdb_data()"

# Clean data
python -c "from app.data.clean import clean_dataset; clean_dataset()"

# Split data
python -c "from app.data.split import split_dataset; split_dataset()"

# Prepare tokenized datasets
python -c "from app.data.tokenizer import prepare_datasets; prepare_datasets()"
```

Or run the complete pipeline:
```bash
python scripts/preprocess_dataset.py
```

## Quality Assurance

- Verify no data leakage between train/val/test sets
- Check label distribution across all sets
- Validate tokenized sequences are within max length
- Ensure no duplicate reviews across splits
- Verify class balance maintained in stratified split

## Data Storage

**Location**: `app/data/`

```
data/
├── raw/
│   └── IMDB.csv
├── processed/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── logs/
    └── pipeline_logs.txt
```

## Future Improvements

- Augmentation techniques (back-translation, paraphrasing)
- Handling of class imbalance with weighted sampling
- Caching of tokenized datasets
- Streaming pipeline for very large datasets
- Data validation schema enforcement
