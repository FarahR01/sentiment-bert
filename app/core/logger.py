"""
Centralized logging configuration.
"""

import logging
import sys


def configure_logging() -> None:
    """
    Configures root logger for the application.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    logging.info("Logging configured successfully.")