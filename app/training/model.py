from transformers import BertForSequenceClassification

def load_model():

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    return model