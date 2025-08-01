from sklearn.metrics import accuracy_score, classification_report
import torch
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.bert_classifier import BertSentimentClassifier

# Load BERT model
model = BertSentimentClassifier()

# Test data
texts = [
    "I love this movie",
    "This film is terrible",
    "Amazing acting and story",
    "Worst movie ever"
]

labels = [1, 0, 1, 0]  # 1=positive, 0=negative

predictions = []
for text in texts:
    pred = model.predict(text)
    # pred returns {'prediction': 0/1, 'confidence': float}
    pred_label = pred['prediction']
    predictions.append(pred_label)

print("Accuracy:", accuracy_score(labels, predictions))
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=['negative', 'positive']))