# API Deployment

## Overview

This document covers deploying the trained sentiment classification model as a REST API using FastAPI, enabling real-time predictions through HTTP endpoints.

## Architecture

### API Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Client Request                        │
│                   (POST /predict)                        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│               FastAPI Application                       │
│            (app/api/main.py)                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │  Routes                                          │   │
│  │  ├─ GET /health                                 │   │
│  │  ├─ POST /predict                               │   │
│  │  ├─ POST /predict_batch                         │   │
│  │  └─ GET /model_info                             │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Input Validation (Pydantic Schemas)            │   │
│  │  ├─ PredictionRequest                           │   │
│  │  └─ BatchPredictionRequest                      │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Prediction Service                             │   │
│  │  (app/services/prediction_service.py)           │   │
│  └──────────────────┬────────────────────────────┘   │
└─────────────────────┼─────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────┐
│            Model & Tokenizer                          │
│                                                        │
│  ├─ BERT Tokenizer (HuggingFace)                     │
│  ├─ Model (PyTorch or ONNX)                          │
│  └─ Inference Engine                                 │
└─────────────────────────────────────────────────────────┘
```

## API Components

### 1. FastAPI Application Setup

**Location**: [`app/api/main.py`](app/api/main.py)

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="Sentiment Analysis API",
    description="BERT-based sentiment classification",
    version="1.0.0"
)

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup"""
    from app.services.prediction_service import load_model
    load_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Request/Response Schemas

**Location**: [`app/api/schemas.py`](app/api/schemas.py)

```python
from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Single prediction request"""
    text: str
    
    class Config:
        example = {
            "text": "This movie was absolutely amazing! Best film I've seen in years."
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str]
    
    class Config:
        example = {
            "texts": [
                "Great movie!",
                "Terrible waste of time."
            ]
        }

class PredictionResponse(BaseModel):
    """Single prediction response"""
    text: str
    prediction: str  # "positive" or "negative"
    confidence: float  # 0.0 to 1.0
    logits: List[float]  # raw model outputs
    
    class Config:
        example = {
            "text": "This movie was amazing!",
            "prediction": "positive",
            "confidence": 0.9823,
            "logits": [0.152, 5.234]
        }

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    results: List[PredictionResponse]
    processing_time_ms: float

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    accuracy: float
    f1_score: float
    model_type: str  # "pytorch" or "onnx"
    max_sequence_length: int
    vocabulary_size: int
```

### 3. Prediction Service

**Location**: [`app/services/prediction_service.py`](app/services/prediction_service.py)

```python
import torch
from transformers import AutoTokenizer
from app.models.bert_classifier import BertSentimentClassifier
from app.core.config import Config

class PredictionService:
    """Service for making predictions"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def load_model(self):
        """Load BERT model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertSentimentClassifier()
        
        # Load trained weights
        checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> dict:
        """Make single prediction"""
        with torch.no_grad():
            # Tokenize input
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Compute probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            
            return {
                'prediction': 'positive' if pred_class == 1 else 'negative',
                'confidence': confidence,
                'logits': logits[0].cpu().tolist(),
                'probabilities': probs[0].cpu().tolist()
            }
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Make batch predictions"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Global service instance
_service = None

def get_prediction_service() -> PredictionService:
    """Get or initialize prediction service"""
    global _service
    if _service is None:
        _service = PredictionService()
        _service.load_model()
    return _service
```

### 4. API Routes

**Location**: [`app/api/main.py`](app/api/main.py)

#### Health Check

```python
@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "message": "API is running",
        "timestamp": datetime.now().isoformat()
    }
```

**Request**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "message": "API is running",
  "timestamp": "2024-03-05T10:30:45.123456"
}
```

#### Single Prediction

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction"""
    service = get_prediction_service()
    result = service.predict(request.text)
    
    return PredictionResponse(
        text=request.text,
        prediction=result['prediction'],
        confidence=result['confidence'],
        logits=result['logits']
    )
```

**Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

**Response**:
```json
{
  "text": "This movie was amazing!",
  "prediction": "positive",
  "confidence": 0.9823,
  "logits": [0.152, 5.234]
}
```

#### Batch Prediction

```python
@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    import time
    start_time = time.time()
    
    service = get_prediction_service()
    results = []
    
    for text in request.texts:
        result = service.predict(text)
        results.append(PredictionResponse(
            text=text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            logits=result['logits']
        ))
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        results=results,
        processing_time_ms=processing_time_ms
    )
```

#### Model Information

```python
@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata"""
    return ModelInfoResponse(
        model_name="bert-base-uncased",
        model_version="1.0.0",
        accuracy=0.9417,
        f1_score=0.9412,
        model_type="pytorch",
        max_sequence_length=512,
        vocabulary_size=30522
    )
```

## Running the API Locally

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install specific packages
pip install fastapi uvicorn torch transformers
```

### Start Server

```bash
# Option 1: Direct Python
python app/api/main.py

# Option 2: Using uvicorn directly
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Use specified port
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

### Access API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Base**: http://localhost:8000

## Docker Deployment

### Build Docker Image

**Location**: [Dockerfile](Dockerfile)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build**:
```bash
docker build -t sentiment-api:latest .
```

**Run**:
```bash
docker run -p 8000:8000 sentiment-api:latest
```

## Production Deployment

### Gunicorn + Uvicorn

```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn app.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

### Cloud Deployment

#### AWS EC2 / Google Cloud / Azure VM

1. Launch VM instance
2. SSH into instance
3. Clone repository
4. Set up environment
5. Run API with process manager (systemd, supervisor)
6. Setup reverse proxy (nginx)

#### Serverless (AWS Lambda, Google Cloud Functions)

Requires API Gateway / Cloud Function wrapper

## Monitoring

### Request Logging

**Location**: `app/api/monitor.py`

```python
from fastapi import Request
from datetime import datetime
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses"""
    import logging
    logger = logging.getLogger("api")
    
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} | "
        f"Time: {process_time:.3f}s"
    )
    
    return response
```

### Metrics (Optional with Prometheus)

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter(
    'predictions_total', 'Total predictions',
    ['status']
)
prediction_latency = Histogram(
    'prediction_latency_seconds', 'Prediction latency'
)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()
```

## Error Handling

```python
from fastapi import HTTPException

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unhandled error: {exc}")
    raise HTTPException(
        status_code=500,
        detail="Internal server error"
    )

# Specific validation error handling
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
```

## Performance Optimization

### Model Optimization

1. **ONNX Export**: Convert to ONNX for faster CPU inference
2. **Quantization**: Reduce model size 4-8x
3. **Batching**: Process multiple requests simultaneously

### API Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(text: str):
    """Cache predictions for identical texts"""
    service = get_prediction_service()
    return service.predict(text)
```

## API Latency

**Expected Performance**:

| Hardware | Latency | Throughput |
|----------|---------|-----------|
| GPU (V100) | 25-50ms | 20-40 req/s |
| GPU (T4) | 50-100ms | 10-20 req/s |
| CPU (8-core) | 100-200ms | 5-10 req/s |

## Testing the API

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "Great movie!"}
)
print(response.json())

# Batch predictions
response = requests.post(
    f"{BASE_URL}/predict_batch",
    json={
        "texts": ["Great!", "Terrible!"]
    }
)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Loved this movie!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Bad."]}'

# Model info
curl http://localhost:8000/model_info
```

## Next Steps

- See [`08_experiment_tracking.md`](08_experiment_tracking.md) for monitoring setup
- See [`09_future_improvements.md`](09_future_improvements.md) for API enhancements
