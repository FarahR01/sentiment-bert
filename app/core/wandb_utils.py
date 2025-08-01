import wandb
import logging
import os

def init_wandb(project_name: str = "nlp-sentiment"):
    """
    Initializes Weights & Biases for experiment tracking.
    """

    try:
        wandb.login()  # Ensure API key is set in env
        wandb.init(project=project_name)
        logging.info("W&B initialized successfully.")
    except Exception as e:
        logging.error(f"W&B initialization failed: {str(e)}")
        raise