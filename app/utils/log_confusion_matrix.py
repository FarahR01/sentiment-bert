from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

def log_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative","Positive"],
        yticklabels=["Negative","Positive"]
    )

    wandb.log({"confusion_matrix": wandb.Image(fig)})