# Project Overview

## Sentiment Analysis with BERT

This project implements a sentiment classification system using BERT (Bidirectional Encoder Representations from Transformers) to classify IMDB movie reviews as positive or negative.

## Project Goals

- Build a robust sentiment classifier that accurately predicts sentiment from text reviews
- Optimize model performance through hyperparameter tuning and training strategies
- Deploy the model as a REST API for real-time predictions
- Track experiments and model metrics using Weights & Biases (W&B)
- Export and serve models in ONNX format for production efficiency

## Key Features

- **BERT-based Classification**: Leverages pre-trained BERT embeddings for higher accuracy
- **Data Pipeline**: Automated ingestion, cleaning, and splitting of IMDB dataset
- **Model Optimization**: ONNX export for faster inference and deployment
- **API Service**: FastAPI-based REST endpoint for predictions
- **Experiment Tracking**: Comprehensive logging via Weights & Biases
- **Monitoring**: Real-time prediction monitoring and error tracking

## Technology Stack

- **Framework**: PyTorch with Transformers library
- **API**: FastAPI
- **Database/Tracking**: Weights & Biases (W&B)
- **Deployment**: Docker, ONNX Runtime
- **Data Processing**: pandas, scikit-learn

## Project Structure

```
sentiment-bert/
├── app/                          # Main application code
│   ├── api/                      # FastAPI routes and schemas
│   ├── core/                     # Configuration and utilities
│   ├── data/                     # Data pipeline modules
│   ├── models/                   # Model definitions and exports
│   ├── services/                 # Prediction services
│   ├── training/                 # Training utilities
│   └── utils/                    # Helper functions
├── scripts/                      # Standalone scripts for training/evaluation
├── tests/                        # Test suite
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
└── Dockerfile                    # Container configuration
```

## Quick Start

1. Set up environment: `conda env create -f environment.yml`
2. Prepare data: `python scripts/preprocess_dataset.py`
3. Train model: `python scripts/train_model.py`
4. Evaluate: `python scripts/evaluate_model.py`
5. Run API: `python app/api/main.py`

## Next Steps

Refer to individual documentation files for detailed information on each component.
