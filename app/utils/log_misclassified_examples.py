import wandb


def log_errors(texts, labels, preds):

    errors = []

    for t, l, p in zip(texts, labels, preds):

        if l != p:

            errors.append({
                "text": t,
                "true": l,
                "pred": p
            })

    wandb.log({"errors": errors[:10]})