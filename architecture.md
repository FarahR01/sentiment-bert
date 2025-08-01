                ┌───────────────────────────────┐
                │        Problem Definition      │
                │  Binary Sentiment Classification│
                │  Labels: Positive / Negative   │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │          Data Collection       │
                │  IMDB Dataset (Labeled Reviews)│
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        Data Preprocessing      │
                │  • Cleaning text               │
                │  • Tokenization (BERT tokenizer)│
                │  • Padding / truncation        │
                │  • Train / Validation split    │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │       Model Selection          │
                │  BERT-base-uncased             │
                │                                │
                │ Alternatives considered:       │
                │  • BiLSTM                      │
                │  • RoBERTa                     │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        Model Training          │
                │  Fine-tuning BERT              │
                │                                │
                │ Key Hyperparameter:            │
                │  learning_rate = 3e-5          │
                │  search range [1e-5,5e-5]      │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │      Experiment Tracking       │
                │  Weights & Biases (W&B)        │
                │                                │
                │ Track:                         │
                │  • Loss curves                 │
                │  • F1 score                    │
                │  • Hyperparameters             │
                │  • Run ID                      │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        Model Evaluation        │
                │                                │
                │ Metrics:                       │
                │  • F1-score                    │
                │  • Precision / Recall          │
                │  • Confusion Matrix            │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        Error Analysis          │
                │                                │
                │ Example failure:               │
                │ "Oh great, another boring movie"│
                │ predicted Positive (sarcasm)   │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │      Overfitting Detection     │
                │                                │
                │ Train F1 = 0.98                │
                │ Val F1   = 0.91                │
                │ → minor overfitting            │
                │ Solution: Dropout 0.1 → 0.3    │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │      Model Optimization        │
                │                                │
                │ Export to ONNX                 │
                │ (faster inference)             │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        API Development         │
                │                                │
                │ FastAPI Endpoint               │
                │ POST /predict                  │
                │                                │
                │ Input: text                    │
                │ Output: sentiment + confidence │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │      Containerization          │
                │                                │
                │ Dockerfile                     │
                │ Reproducible environment       │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        Deployment              │
                │                                │
                │ FastAPI + Uvicorn              │
                │ Docker Container               │
                │ REST API                       │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │     Latency Measurement        │
                │                                │
                │ ~10–15 ms per request          │
                │ (GPU inference)                │
                └───────────────────────────────┘