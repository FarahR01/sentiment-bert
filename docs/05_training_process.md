# Training Process

## Overview

This document details the complete model training pipeline, including setup, hyperparameter configuration, training loop, validation, and checkpoint management.

## Training Script

**Location**: [`scripts/train_model.py`](scripts/train_model.py)

Execute training with:
```bash
python scripts/train_model.py
```

## Training Pipeline

### Phase 1: Initialization

```
Setup
  ├─ Load configuration (config.py)
  ├─ Initialize logger
  ├─ Set random seeds (reproducibility)
  ├─ Configure device (GPU/CPU)
  ├─ Load tokenizer
  └─ Prepare datasets

Load Data
  ├─ Load train.csv
  ├─ Load val.csv
  ├─ Create PyTorch DataLoaders
  └─ Verify class distribution

Initialize Model
  ├─ Load BERT base uncased
  ├─ Add classification head
  ├─ Load pre-trained weights
  └─ Move to device
```

### Phase 2: Training Setup

**Hyperparameters**:

```python
LEARNING_RATE = 2e-5          # Lower for fine-tuning
BATCH_SIZE = 32               # GPU memory efficient
NUM_EPOCHS = 3                # Usually sufficient for BERT
WEIGHT_DECAY = 0.01           # L2 regularization
WARMUP_RATIO = 0.1            # Gradually increase LR
GRADIENT_CLIP = 1.0           # Gradient clipping
```

**Optimizer Configuration**:

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8
)
```

**Learning Rate Scheduler**:

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### Phase 3: Training Loop

```
for epoch in 1..NUM_EPOCHS:
    
    Training Phase:
    ├─ Set model to train mode
    ├─ Initialize metrics
    └─ for batch in train_loader:
       ├─ Forward pass: logits = model(input_ids, attention_mask)
       ├─ Compute loss: loss = criterion(logits, labels)
       ├─ Backward pass: loss.backward()
       ├─ Gradient clipping: clip_grad_norm_()
       ├─ Optimizer step: optimizer.step()
       ├─ Learning rate step: scheduler.step()
       ├─ Update metrics (loss, accuracy)
       └─ Log to W&B
    
    Validation Phase:
    ├─ Set model to eval mode
    ├─ Initialize metrics
    └─ for batch in val_loader:
       ├─ Forward pass (no grad)
       ├─ Compute loss
       ├─ Compute predictions
       ├─ Update metrics (loss, accuracy, F1)
       └─ Accumulate predictions
    
    Post-Epoch:
    ├─ Compute validation metrics (Precision, Recall, F1)
    ├─ Log metrics to W&B
    ├─ Compare with best validation loss
    ├─ If improved:
    │  ├─ Save checkpoint
    │  └─ Store best weights
    └─ Print epoch summary
```

### Phase 4: Checkpoint Management

**Saved During Training**:

```
checkpoints/
├── checkpoint-epoch-1/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── checkpoint-epoch-2/
├── checkpoint-epoch-3/
└── best_model.pt
```

**Best Model Selection**:
- Monitored metric: Validation Loss (lower is better)
- Alternative: Validation F1-Score (higher is better)
- Saves weights when validation metric improves

## Training Metrics

### Per-Batch Metrics
- **Loss**: Cross-entropy loss
- **Batch Accuracy**: Proportion of correct predictions
- **Learning Rate**: Current LR from scheduler

### Per-Epoch Metrics
- **Train Loss**: Average training loss
- **Train Accuracy**: Overall training accuracy
- **Val Loss**: Validation set loss
- **Val Accuracy**: Validation set accuracy
- **Val Precision**: True positives / (true + false positives)
- **Val Recall**: True positives / (true positives + false negatives)
- **Val F1-Score**: Harmonic mean of precision and recall
- **Val ROC-AUC**: Area under ROC curve

### Metric Computation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
auc = roc_auc_score(true_labels, probabilities)
```

## Training Monitoring

### Weights & Biases Integration

**Configuration**: `app/core/wandb_utils.py`

**Tracked Metrics**:
- Training and validation loss
- Learning rate scheduling
- Model accuracy and F1-score
- Confusion matrices
- Sample predictions
- Hardware utilization (GPU memory, training time)

**Logging**:
```python
import wandb

wandb.init(project="sentiment-bert", entity="your-entity")

for epoch in range(NUM_EPOCHS):
    # ... training ...
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'learning_rate': current_lr
    })

wandb.finish()
```

### Console Output

Example training log:
```
Epoch 1/3
Train Loss: 0.425 | Train Acc: 0.815 | Val Loss: 0.328 | Val Acc: 0.872 | Val F1: 0.869
Epoch 2/3
Train Loss: 0.245 | Train Acc: 0.910 | Val Loss: 0.289 | Val Acc: 0.893 | Val F1: 0.891
Epoch 3/3
Train Loss: 0.156 | Train Acc: 0.945 | Val Loss: 0.298 | Val Acc: 0.899 | Val F1: 0.898
```

## Regularization Techniques

### Dropout
- Applied in BERT embeddings and classification head
- Probability: 0.1 (10% of neurons dropped)
- Prevents overfitting

### Weight Decay
- L2 regularization via AdamW optimizer
- Value: 0.01
- Penalizes large weights

### Gradient Clipping
- Maximum norm: 1.0
- Prevents exploding gradients during backprop

### Early Stopping (Optional)
- Monitor validation loss
- Stop if no improvement for N epochs
- Prevents overfitting

## Device Management

### GPU Training
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Batch tensors moved to device
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)
```

### Distributed Training (Optional)
- DataParallel for multiple GPUs
- DistributedDataParallel for multi-node

## Training Duration

**Typical Training Time**:
- Per epoch: 5-10 minutes (GPU, 25k samples)
- Total (3 epochs): 15-30 minutes
- CPU training: 5-10x slower

## Reproducibility

**Fixed Random Seeds**:
```python
import random
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory | Batch too large | Reduce BATCH_SIZE |
| NaN loss | Gradient explosion | Reduce learning rate, add gradient clipping |
| No improvement | Model underfitting | Increase epochs, increase model capacity |
| Slow convergence | Learning rate too low | Increase LEARNING_RATE |
| Overfitting | Model too complex | Add dropout, reduce epochs, use L2 regularization |

## Post-Training Steps

1. **Evaluation**: Run evaluation script (see `06_evaluation_and_error_analysis.md`)
2. **Model Export**: Convert to ONNX for deployment
3. **Documentation**: Update model card with final metrics
4. **Archival**: Store best model and training logs

## Next Steps

- See [`06_evaluation_and_error_analysis.md`](06_evaluation_and_error_analysis.md) for test set evaluation
- See [`08_experiment_tracking.md`](08_experiment_tracking.md) for detailed W&B setup
