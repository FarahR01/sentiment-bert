# Model Architecture

## Overview

This project uses a fine-tuned BERT-based architecture for binary sentiment classification. The model leverages pre-trained contextual embeddings combined with a classification head.

## BERT Base Architecture

### Architecture Components

```
Input Text
    ↓
[Tokenization & Embedding]
    ├─ Token Embeddings (WordPiece)
    ├─ Positional Embeddings
    └─ Segment Embeddings
    ↓
[BERT Encoder Stack]
    ├─ 12 Transformer Blocks
    ├─ 12 Attention Heads per block
    ├─ 768-dimensional hidden state
    └─ 110M parameters (full)
    ↓
[Classification Head]
    ├─ [CLS] Token Representation (768-dim)
    ├─ Dropout (p=0.1)
    ├─ Dense Layer → 512 neurons
    ├─ ReLU Activation
    ├─ Dropout (p=0.1)
    └─ Output Layer → 2 neurons (Logits)
    ↓
[Softmax + Cross-Entropy Loss]
    ↓
Sentiment Prediction (Positive/Negative)
```

## Model Specifications

### BERT Base Uncased

- **Model Name**: `bert-base-uncased`
- **Source**: HuggingFace Transformers
- **Parameters**: ~110 million
- **Hidden Size**: 768
- **Num Hidden Layers**: 12
- **Num Attention Heads**: 12
- **Intermediate Size**: 3072
- **Vocab Size**: 30,522
- **Max Position Embeddings**: 512
- **Type Vocab Size**: 2

### Configuration

```python
from transformers import BertConfig

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
)
```

## Custom Classification Layer

### BertForSentimentClassification

Built on top of pre-trained BERT with custom classification head:

**Layer Details**:

1. **BERT Encoder**: Pre-trained, optionally frozen
   - Processes input tokens through 12 transformer blocks
   - Outputs contextualized embeddings

2. **Pooling**: Extract [CLS] token
   - Special token at sequence start
   - Represents entire sequence in single vector
   - Output shape: (batch_size, 768)

3. **Dropout**: Regularization (10%)
   - Prevents overfitting to training data
   - Applied during training only

4. **Dense Layer**: 768 → 512 neurons
   - Reduces dimensionality
   - Captures sentiment-specific features
   - Activation: ReLU

5. **Dropout**: Regularization (10%)
   - Second regularization layer

6. **Output Layer**: 512 → 2 neurons
   - Raw logits for positive/negative classes
   - No activation (applied during inference via softmax)

**Output**: Logits (batch_size, 2) → contains scores for each class

## Implementation

**Location**: [`app/models/bert_classifier.py`](app/models/bert_classifier.py)

```python
from transformers import AutoModel, AutoTokenizer

class BertSentimentClassifier(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output  # [CLS] representation
        logits = self.classifier(pooled_output)
        return logits
```

## Training Strategy

### Loss Function
- **Loss**: Cross-Entropy Loss (BCEWithLogitsLoss for binary classification)
- Measures difference between predicted logits and true labels
- Handles class imbalance through optional weighting

### Optimization
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 (typical for BERT fine-tuning)
- **Weight Decay**: 0.01
- **Warmup Steps**: 500 (builds up learning rate gradually)

### Hyperparameters
```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VALUE = 1.0
```

## Model Variants

### Memory-Efficient Variants

For deployment with limited resources:

1. **BERT Distilled** (DistilBERT)
   - 40% fewer parameters
   - 60% faster
   - 97% performance retention

2. **ONNX Quantized**
   - 4x smaller model
   - Faster CPU inference
   - Minimal accuracy loss

## Model Outputs

### Training/Validation
- **Logits**: Raw scores (batch_size, 2)
- **Loss**: Scalar value (backpropagated)

### Inference
- **Logits**: Raw scores
- **Probabilities**: Softmax(logits) → (0, 1)
- **Predictions**: argmax(softmax) → {0: Negative, 1: Positive}
- **Confidence**: max(softmax) → probability of predicted class

## Model Files

**Location**: [`app/models/`](app/models/)

- `bert_classifier.py`: Model architecture
- `test_model.pt`: PyTorch saved model
- `bert_sentiment.onnx`: ONNX exported model (optimized for inference)

## Inference

### PyTorch Inference
```python
model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
```

### ONNX Inference (Production)
- Faster CPU inference using ONNX Runtime
- Lower memory footprint
- No PyTorch dependency required

## Performance Considerations

### Memory
- Model + optimizer: ~1.2GB (training)
- Model only: ~440MB (inference with PyTorch)
- ONNX quantized: ~110MB

### Speed
- Forward pass: ~50-100ms per sample (GPU)
- Training batch: ~500ms for 32 samples

### Accuracy
- Target F1-Score: >0.90
- Expected on IMDB: ~93% accuracy

## Fine-tuning Strategies

1. **Full Fine-tuning**: Update all BERT weights
2. **Layer-wise: Fine-tune last N layers only
3. **LoRA**: Low-rank adaptation for fewer parameters
4. **Prompt-based**: Uses prompts instead of full fine-tuning

## Next Steps

See model training details in [`05_training_process.md`](05_training_process.md)
