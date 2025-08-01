"""
Reproducibility utilities.
Controls randomness across Python, NumPy, and PyTorch.
"""

import torch
import numpy as np
import random
import logging


def set_global_seed(seed: int = 42) -> None:
    """
    Sets global seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logging.info(f"Global seed set to {seed}")

    except Exception as e:
        logging.error(f"Failed to set global seed: {str(e)}")
        raise