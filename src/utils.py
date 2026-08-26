"""
utils.py

Shared utilities for the FinShield Fraud Detection Platform: project-root
resolution, logging setup, and model metadata loading.

Every notebook so far resolved the project root with its own
`Path.cwd().resolve().parent` + `sys.path.append(...)` hack, which is
fragile because it silently depends on the directory Jupyter happens to be
started from. Scripts and the API need the same root but run with a
different cwd again. Centralizing it here means there is exactly one place
that knows where the project lives.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger, safe to call repeatedly (e.g. on module
    reload in a notebook) without accumulating duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_model_metadata(model_dir: Path = MODEL_DIR) -> dict:
    """
    Load the metadata written by src.train (winning model name, tuned
    hyperparameters, decision threshold, CV comparison, test metrics).
    """
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Run `python -m src.train` first to train and save a model."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
