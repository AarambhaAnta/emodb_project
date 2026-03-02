"""PLDA training for emotion recognition using plda.Classifier.

https://github.com/RaviSoji/plda

Workflow:
  1. Load embeddings + labels from the embeddings CSV produced by extract_embeddings.py
  2. Fit plda.Classifier on the training split
  3. Save as plda_model.pkl per speaker
"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from plda import Classifier

logger = logging.getLogger(__name__)


def _load_embeddings(emb_csv):
    """Load (X, y) from an embeddings CSV."""
    df = pd.read_csv(emb_csv)
    X, y = [], []
    for _, row in df.iterrows():
        path = str(row['embedding'])
        if not os.path.exists(path):
            logger.warning(f"Missing embedding file: {path}")
            continue
        emb = np.load(path).flatten().astype(np.float64)
        emb = np.nan_to_num(emb)
        X.append(emb)
        y.append(int(row['label']))
    if not X:
        raise ValueError(f"No embeddings loaded from {emb_csv}")
    X = np.array(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0), np.array(y)


class PLDAModel:
    """Wrapper around plda.Classifier for save/load and predict_proba."""

    def __init__(self, clf=None):
        self.clf = clf

    def fit(self, X, y, n_components=None):
        self.clf = Classifier()
        self.clf.fit_model(X, y, n_principal_components=n_components)
        return self

    def predict(self, X):
        preds, _ = self.clf.predict(X)
        return preds

    def predict_proba(self, X):
        _, logps = self.clf.predict(X)
        logps -= logps.max(axis=1, keepdims=True)
        probs = np.exp(logps)
        return probs / probs.sum(axis=1, keepdims=True)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_speaker_plda(speaker_id, loso_dir, output_dir, embeddings_dir,
                       embeddings_split='train', n_components=None):
    """Train a PLDA classifier for one speaker and save it.

    Args:
        speaker_id: e.g. '03'
        loso_dir: base LOSO directory (kept for API consistency)
        output_dir: where to write plda_model.pkl
        embeddings_dir: base embeddings directory
        embeddings_split: split to train on (default 'train'); also loads 'val' automatically
        n_components: PLDA latent dimensionality (None = auto)

    Returns:
        dict with 'model_path' and 'n_train_samples'
    """
    emb_spk = os.path.join(embeddings_dir, f"speaker_{speaker_id}")
    emb_csv = os.path.join(emb_spk, f"{embeddings_split}_embeddings.csv")

    if not os.path.exists(emb_csv):
        raise FileNotFoundError(f"Embeddings CSV not found: {emb_csv}")

    X, y = _load_embeddings(emb_csv)

    # Also include val embeddings if available (more data → better PLDA)
    val_csv = os.path.join(emb_spk, "val_embeddings.csv")
    if embeddings_split == 'train' and os.path.exists(val_csv):
        X_val, y_val = _load_embeddings(val_csv)
        X = np.concatenate([X, X_val], axis=0)
        y = np.concatenate([y, y_val], axis=0)
        logger.info(f"Speaker {speaker_id}: added val embeddings ({len(X_val)} samples)")

    logger.info(f"Speaker {speaker_id}: training PLDA on {len(X)} samples, "
                f"{len(np.unique(y))} classes")

    model = PLDAModel().fit(X, y, n_components=n_components)

    out_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
    model_path = os.path.join(out_dir, "plda_model.pkl")
    model.save(model_path)

    logger.info(f"Speaker {speaker_id}: PLDA saved → {model_path}")
    return {'model_path': model_path, 'n_train_samples': len(X), 'speaker': speaker_id}


def train_all_speakers_plda(loso_dir, output_dir, embeddings_dir,
                             embeddings_split='train', n_components=None):
    """Train PLDA for all speakers; returns per-speaker results dict."""
    os.makedirs(output_dir, exist_ok=True)
    speakers = sorted(
        d.replace('speaker_', '') for d in os.listdir(loso_dir)
        if d.startswith('speaker_')
    )
    results = {}
    for spk in tqdm(speakers, desc="Training PLDA"):
        try:
            results[spk] = train_speaker_plda(
                spk, loso_dir, output_dir, embeddings_dir,
                embeddings_split=embeddings_split, n_components=n_components,
            )
        except Exception as e:
            logger.error(f"Speaker {spk} PLDA failed: {e}")
            results[spk] = {'status': 'failed', 'error': str(e)}
    return results
