"""Sentiment scoring.

Real mode: FinBERT (ProsusAI/finbert) via HuggingFace transformers.
Demo mode: uses pre-labeled `true_sentiment` column from the synthetic news
           generator, with realistic noise added to simulate model error.

Both modes produce the same output schema so downstream code is source-agnostic.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def score_news_demo(news_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Demo scorer — adds realistic noise to pre-labeled sentiment.

    Outputs: positive, negative, neutral probabilities + `compound` score in [-1, 1].
    """
    rng = np.random.default_rng(seed + 2)
    df = news_df.copy()

    # Add ~10% gaussian noise to simulate FinBERT's imperfection
    noisy = df["true_sentiment"].values + rng.normal(0, 0.1, len(df))
    compound = np.clip(noisy, -1, 1)

    # Convert to 3-class probabilities
    pos = np.where(compound > 0, np.abs(compound) * 0.8 + 0.1, 0.1)
    neg = np.where(compound < 0, np.abs(compound) * 0.8 + 0.1, 0.1)
    neu = 1.0 - pos - neg
    neu = np.clip(neu, 0.05, 1.0)

    # Normalize
    total = pos + neg + neu
    df["sent_positive"] = pos / total
    df["sent_negative"] = neg / total
    df["sent_neutral"]  = neu / total
    df["sent_compound"] = compound
    return df


def score_news_real(news_df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Score headlines with FinBERT.

    Runs on GPU if available, otherwise CPU (slower but works).
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError("Install torch + transformers: pip install torch transformers") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading FinBERT on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL).to(device)
    model.eval()

    # FinBERT label order: [positive, negative, neutral]
    id2label = model.config.id2label  # e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}

    headlines = news_df["headline"].fillna("").tolist()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    df = news_df.copy()

    # Map by label name so we're robust to index order
    label_to_col = {"positive": "sent_positive", "negative": "sent_negative", "neutral": "sent_neutral"}
    for idx in range(probs.shape[1]):
        label = id2label[idx]
        df[label_to_col[label]] = probs[:, idx]

    df["sent_compound"] = df["sent_positive"] - df["sent_negative"]
    return df


def score_news(news_df: pd.DataFrame, demo: bool = True) -> pd.DataFrame:
    if demo:
        return score_news_demo(news_df)
    return score_news_real(news_df)
