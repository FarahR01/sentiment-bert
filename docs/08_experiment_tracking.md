# Experiment Tracking

## Overview

This document describes the experiment tracking infrastructure using Weights & Biases (W&B) for monitoring training, evaluation, and production performance.

## Weights & Biases Integration

### Setup

**Location**: [`app/core/wandb_utils.py`](app/core/wandb_utils.py)

```python
import wandb
from app.core.config import Config

def init_wandb(project_name: str, entity: str = None):
    """Initialize Weights & Biases tracking"""
    wandb.init(
        project=project_name,
        entity=entity,  # Optional: your W&B team
        name=f"sentiment-bert-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["bert", "sentiment", "imdb"],
        notes="BERT sentiment classification on IMDB dataset"
    )
    
    # Log config
    wandb.config.update({
        "learning_rate": Config.LEARNING_RATE,
        "batch_size": Config.BATCH_SIZE,
        "num_epochs": Config.NUM_EPOCHS,
        "model": "bert-base-uncased"
    })

def finish_wandb():
    """Finish W&B run"""
    wandb.finish()
```

### Installation

```bash
pip install wandb
```

### Authentication

```bash
wandb login
# Paste API key from https://wandb.ai/settings/api
```

## Training Metrics Logging

### Per-Epoch Metrics

```python
for epoch in range(NUM_EPOCHS):
    # ... training loop ...
    
    # Log metrics at end of epoch
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss / num_batches,
        'train_accuracy': train_correct / train_total,
        'val_loss': val_loss / num_val_batches,
        'val_accuracy': val_correct / val_total,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1_score': val_f1,
        'val_roc_auc': val_auc,
        'learning_rate': current_lr,
    })
```

### Per-Batch Metrics

```python
for batch_idx, batch in enumerate(train_loader):
    # ... forward pass ...
    
    # Log every 100 batches
    if batch_idx % 100 == 0:
        wandb.log({
            'batch': batch_idx,
            'batch_loss': loss.item(),
            'batch_accuracy': batch_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        })
```

### Example Training Run

```
Epoch 1/3 | Step 100
  batch_loss: 0.598
  val_loss: 0.385
  val_accuracy: 0.845
  
Epoch 2/3 | Step 200
  batch_loss: 0.276
  val_loss: 0.289
  val_accuracy: 0.901

Epoch 3/3 | Step 300
  batch_loss: 0.142
  val_loss: 0.298
  val_accuracy: 0.912
```

## Evaluation Metrics

### Logging Evaluation Results

```python
from app.utils.log_confusion_matrix import log_confusion_matrix_to_wandb

# After evaluation
wandb.log({
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1_score': test_f1,
    'test_roc_auc': test_auc,
    'test_specificity': test_specificity,
})

# Log confusion matrix visualization
log_confusion_matrix_to_wandb(
    true_labels=test_labels,
    predictions=test_predictions,
    class_names=['negative', 'positive']
)
```

### Classification Report

```python
from sklearn.metrics import classification_report

report = classification_report(
    test_labels, test_predictions,
    output_dict=False
)

# Log as artifact
with open("classification_report.txt", "w") as f:
    f.write(report)

wandb.save("classification_report.txt")
```

## Confusion Matrix Logging

**Location**: [`app/utils/log_confusion_matrix.py`](app/utils/log_confusion_matrix.py)

```python
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np

def log_confusion_matrix_to_wandb(true_labels, predictions, class_names):
    """Log confusion matrix to W&B"""
    cm = confusion_matrix(true_labels, predictions)
    
    # Create W&B confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=true_labels,
            preds=predictions,
            class_names=class_names,
            title="Confusion Matrix"
        )
    })
    
    # Log raw values
    wandb.log({
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1]
    })
```

## Misclassified Examples Logging

**Location**: [`app/utils/log_misclassified_examples.py`](app/utils/log_misclassified_examples.py)

```python
import wandb

def log_misclassified_examples(texts, true_labels, predictions, probs):
    """Log misclassified examples to W&B"""
    false_positives = []
    false_negatives = []
    
    for text, true, pred, prob in zip(texts, true_labels, predictions, probs):
        if true != pred:
            example = {
                'text': text,
                'true_label': class_names[true],
                'predicted_label': class_names[pred],
                'confidence': prob[pred],
            }
            
            if pred == 1 and true == 0:  # FP
                false_positives.append(example)
            elif pred == 0 and true == 1:  # FN
                false_negatives.append(example)
    
    # Log tables
    wandb.log({
        'false_positives': wandb.Table(dataframe=pd.DataFrame(false_positives[:10])),
        'false_negatives': wandb.Table(dataframe=pd.DataFrame(false_negatives[:10])),
    })
```

## Model Artifacts

### Saving Model Checkpoints

```python
import os

def save_checkpoint(model, optimizer, epoch, metrics):
    """Save model checkpoint to W&B"""
    
    # Local save
    checkpoint_path = f'checkpoints/epoch-{epoch}.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    # Log to W&B
    wandb.save(checkpoint_path)
    
    # Log metrics
    wandb.log({
        'checkpoint_epoch': epoch,
        'saved_metrics': metrics
    })
```

### Best Model Artifact

```python
def save_best_model(model, metrics):
    """Save best model as W&B artifact"""
    
    # Save locally
    torch.save(model.state_dict(), 'best_model.pt')
    
    # Create artifact
    artifact = wandb.Artifact('best-sentiment-model', type='model')
    artifact.add_file('best_model.pt')
    
    # Add metadata
    artifact.metadata = {
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'model_type': 'bert-base-uncased'
    }
    
    wandb.log_artifact(artifact)
```

## System Metrics

### Hardware Monitoring

W&B automatically logs:
- GPU memory usage
- GPU utilization
- CPU usage
- Memory consumption
- Training time

**Example System Metrics**:
```
GPU Memory Used: 4.2 / 16 GB
GPU Utilization: 85%
CPU Usage: 45%
Training Time: 8h 34m
```

## Experiment Configuration

### Logging Hyperparameters

```python
config = {
    # Model
    'model_name': 'bert-base-uncased',
    'num_classes': 2,
    'hidden_size': 768,
    
    # Training
    'learning_rate': 2e-5,
    'batch_size': 32,
    'num_epochs': 3,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    
    # Data
    'dataset': 'IMDB',
    'train_size': 17500,
    'val_size': 3750,
    'test_size': 3750,
    'max_seq_length': 512,
    
    # Regularization
    'dropout': 0.1,
    'gradient_clip': 1.0,
}

wandb.config.update(config)
```

## Experiment Comparison

### View on W&B Dashboard

1. Go to https://wandb.ai/your-username/sentiment-bert
2. Click "Runs" tab
3. Select multiple runs to compare
4. View parallel coordinates or tables

### Example Comparison

| Run | LR | BS | Epochs | Val Acc | Val F1 | Notes |
|-----|----|----|--------|---------|--------|-------|
| run-1 | 2e-5 | 32 | 3 | 0.890 | 0.887 | baseline |
| run-2 | 5e-5 | 32 | 3 | 0.875 | 0.870 | higher LR |
| run-3 | 2e-5 | 16 | 4 | 0.899 | 0.896 | smaller BS |

## Custom Charts

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    true_labels, 
    probabilities[:, 1]  # probabilities for positive class
)

wandb.log({
    'precision_recall_curve': wandb.plot.line_series(
        x=recall,
        y=[precision],
        title='Precision-Recall Curve',
        xname='Recall',
        yname='Precision'
    )
})
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
roc_auc = auc(fpr, tpr)

wandb.log({
    'roc_curve': wandb.plot.line_series(
        x=fpr,
        y=[tpr],
        title=f'ROC Curve (AUC={roc_auc:.3f})',
        xname='False Positive Rate',
        yname='True Positive Rate'
    )
})
```

## Reports

### Creating a Report

1. Go to W&B project page
2. Click "Reports" → "Create Report"
3. Add panels:
   - Training loss over time
   - Validation metrics
   - Confusion matrix
   - Best vs worst predictions
   - Hyperparameter comparison

### Report Elements

```
┌─────────────────────────────────────────┐
│        Training Summary Report          │
├─────────────────────────────────────────┤
│ • Project: sentiment-bert               │
│ • Best Model Accuracy: 94.17%           │
│ • Best Model F1-Score: 94.12%           │
│                                         │
│ [Loss Curve]  [Accuracy Curve]         │
│ [Confusion]   [ROC Curve]               │
│ [LR Schedule] [Metrics Table]           │
└─────────────────────────────────────────┘
```

## Production Monitoring

### Logging Predictions

```python
def log_production_prediction(text, prediction, confidence):
    """Log production predictions for monitoring"""
    wandb.log({
        'production_text': text,
        'production_prediction': prediction,
        'production_confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })
```

### Monitoring Drift

```python
def check_prediction_drift(recent_predictions, historical_predictions):
    """Monitor for prediction distribution changes"""
    
    recent_positive_rate = sum(
        1 for p in recent_predictions if p == 'positive'
    ) / len(recent_predictions)
    
    historical_positive_rate = sum(
        1 for p in historical_predictions if p == 'positive'
    ) / len(historical_predictions)
    
    drift = abs(recent_positive_rate - historical_positive_rate)
    
    wandb.log({
        'prediction_drift': drift,
        'recent_positive_rate': recent_positive_rate,
        'historical_positive_rate': historical_positive_rate,
        'alert': drift > 0.05  # Alert if >5% change
    })
```

## Sweeps (Hyperparameter Tuning)

### Create Sweep Config

**sweep.yaml**:
```yaml
program: scripts/train_model.py
method: bayes  # grid, random, bayes
metric:
  name: val_f1_score
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform
    min: 1e-6
    max: 1e-3
  batch_size:
    values: [16, 32, 64]
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
```

### Run Sweep

```bash
wandb sweep sweep.yaml
# Output: sweep ID
wandb agent username/project/sweep-id
```

## Integration with Scripts

### Training Script Integration

**scripts/train_model.py**:
```python
from app.core.wandb_utils import init_wandb, finish_wandb

init_wandb("sentiment-bert")

try:
    # Training loop with wandb.log() calls
    for epoch in range(num_epochs):
        # ... training ...
        wandb.log({'epoch': epoch, ...})
finally:
    finish_wandb()
```

### Evaluation Script Integration

**scripts/evaluate_model.py**:
```python
from app.core.wandb_utils import init_wandb

init_wandb("sentiment-bert-eval")

# Load best model from W&B
artifact = wandb.use_artifact('sentiment-bert/best-sentiment-model:latest')
model_path = artifact.get_path('best_model.pt').download()

# ... evaluation ...
wandb.log({'test_accuracy': test_acc, ...})
wandb.finish()
```

## W&B API Usage

### Querying Experiments

```python
import wandb

api = wandb.Api()

# Get specific run
run = api.run("username/project/run-id")
print(run.summary)  # Final metrics

# Get all runs
runs = api.runs("username/project")
for run in runs:
    print(f"{run.name}: {run.summary['val_f1_score']}")
```

### Downloading Artifacts

```python
api = wandb.Api()
artifact = api.artifact('username/project/model:latest')
artifact_dir = artifact.download()
```

## W&B Dashboard Features

- **Real-time Monitoring**: Watch training live
- **Experiment Comparison**: Side-by-side metric comparison
- **Reports**: Markdown documents with embedded metrics
- **Alerts**: Notifications for training issues
- **Collaboration**: Share runs and reports with team

## Best Practices

1. **Name Runs Descriptively**: Include hyperparameters in run name
2. **Log Frequently**: Log every epoch or every N batches
3. **Version Models**: Use artifacts for model versioning
4. **Document Experiments**: Add notes explaining run goals
5. **Create Reports**: Document findings in W&B reports
6. **Monitor Production**: Track prediction drift and confidence
7. **Organize Projects**: Group related experiments in same project

## Next Steps

- See [`06_evaluation_and_error_analysis.md`](06_evaluation_and_error_analysis.md) for detailed metrics
- See [`09_future_improvements.md`](09_future_improvements.md) for enhancements
