"""PLDA scoring: predict emotion labels for test embeddings and compute metrics."""
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from utils.extract_config import get_config
from utils.training.plda_trainer import PLDAModel

EMOTION_LABELS = {
    0: "Happiness", 1: "Neutral", 2: "Anger",
    3: "Fear",      4: "Boredom", 5: "Disgust", 6: "Sadness",
}


def _l2_norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _load_test_embeddings(emb_csv):
    """Load raw (X, y_true, ids) from a *_embeddings.csv file."""
    df = pd.read_csv(emb_csv)
    X, y, ids = [], [], []
    for _, row in df.iterrows():
        path = str(row['embedding'])
        if not os.path.exists(path):
            continue
        emb = np.load(path).flatten().astype(np.float64)
        emb = np.nan_to_num(emb)
        X.append(emb)
        y.append(int(row['label']))
        ids.append(str(row.get('ID', row.get('id', ''))))
    return np.array(X), np.array(y), ids


def _aggregate_utterances(X, y, ids):
    """Average segments per utterance (strip trailing _N), then L2-normalise.

    Returns utterance-level (X_agg, y_agg, ids_agg).
    """
    base_ids = [re.sub(r'_\d+$', '', sid) for sid in ids]
    groups = defaultdict(list)
    label_for = {}
    for i, (base_id, label) in enumerate(zip(base_ids, y)):
        groups[base_id].append(i)
        label_for[base_id] = label

    X_agg, y_agg, ids_agg = [], [], []
    for base_id in sorted(groups):
        idx = groups[base_id]
        X_agg.append(X[idx].mean(axis=0))
        y_agg.append(label_for[base_id])
        ids_agg.append(base_id)

    return _l2_norm(np.array(X_agg)), np.array(y_agg), ids_agg


def _plot(results_dir, conf_mat, report):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if conf_mat is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(conf_mat, cmap='Blues')
            labels = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)]
            ax.set_xticks(range(7)); ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(range(7)); ax.set_yticklabels(labels)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            fig.tight_layout()
            fig.savefig(results_dir / 'confusion_matrix.png', dpi=150)
            plt.close(fig)

        if report:
            names = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS) if EMOTION_LABELS[i] in report]
            f1s = [report[n]['f1-score'] for n in names]
            if f1s:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(names, f1s, color='#4C78A8')
                ax.set_title('F1 Score per Emotion'); ax.set_ylabel('F1'); ax.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(results_dir / 'f1_per_class.png', dpi=150)
                plt.close(fig)
    except Exception:
        pass  # plots are optional


def score_speaker_plda(speaker_id, base_dir=None, results_dir=None):
    """Score test embeddings for one speaker using the trained PLDA model.

    Returns a dict with 'accuracy', 'f1_macro', 'f1_weighted', 'scores_csv'.
    """
    config = get_config()
    base_dir = Path(base_dir or config['BASE_DIR'])
    paths = config.get('PATHS', {})

    emb_dir = base_dir / paths.get('EMBEDDINGS', 'output/embeddings') / f"speaker_{speaker_id}"
    model_path = base_dir / 'output' / 'models' / 'plda' / f"speaker_{speaker_id}" / 'plda_model.pkl'
    out_dir = Path(results_dir or (base_dir / paths.get('RESULTS', 'output/results'))) / f"speaker_{speaker_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"PLDA model not found: {model_path}")

    test_csv = emb_dir / 'test_embeddings.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

    model = PLDAModel.load(str(model_path))
    X_test, y_true, ids = _load_test_embeddings(str(test_csv))
    X_test, y_true, ids = _aggregate_utterances(X_test, y_true, ids)

    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    accuracy   = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_micro   = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat   = confusion_matrix(y_true, y_pred, labels=sorted(EMOTION_LABELS))
    report     = classification_report(
        y_true, y_pred,
        labels=sorted(EMOTION_LABELS),
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)],
        zero_division=0, output_dict=True,
    )

    # Per-sample scores CSV
    rows = []
    for i, (sid, gt, pred) in enumerate(zip(ids, y_true, y_pred)):
        prob_row = {EMOTION_LABELS[j]: float(f"{probs[i, j]:.5f}")
                   for j in range(len(EMOTION_LABELS))}
        rows.append({'id': sid, 'ground_truth': int(gt), 'predicted': int(pred),
                     'correct': int(gt) == int(pred), **prob_row})
    scores_csv = out_dir / 'plda_scores.csv'
    pd.DataFrame(rows).to_csv(scores_csv, index=False)

    metrics = {
        'speaker': speaker_id, 'accuracy': accuracy,
        'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted,
        'scores_csv': str(scores_csv),
        'classification_report': report,
        'confusion_matrix': conf_mat.tolist(),
    }
    with open(out_dir / 'plda_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    _plot(out_dir, conf_mat, report)
    return metrics


def score_all_speakers_plda(base_dir=None, results_dir=None):
    """Score all speakers; returns per-speaker metrics dict and writes a summary CSV."""
    config = get_config()
    base_dir = Path(base_dir or config['BASE_DIR'])
    emb_root = base_dir / config.get('PATHS', {}).get('EMBEDDINGS', 'output/embeddings')

    speakers = sorted(
        d.replace('speaker_', '') for d in os.listdir(emb_root)
        if d.startswith('speaker_')
    ) if emb_root.exists() else []

    all_results = {}
    for spk in speakers:
        try:
            all_results[spk] = score_speaker_plda(spk, base_dir=base_dir, results_dir=results_dir)
        except Exception as e:
            all_results[spk] = {'error': str(e)}

    # Summary CSV
    out_root = Path(results_dir or (base_dir / config.get('PATHS', {}).get('RESULTS', 'output/results')))
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [{'speaker': spk, 'accuracy': r.get('accuracy'), 'f1_macro': r.get('f1_macro'),
             'f1_weighted': r.get('f1_weighted'), 'error': r.get('error')}
            for spk, r in all_results.items()]
    pd.DataFrame(rows).to_csv(out_root / 'plda_summary.csv', index=False)

    return all_results
