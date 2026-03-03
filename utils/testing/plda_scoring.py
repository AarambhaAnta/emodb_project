"""Score test speaker using ECAPA embeddings + PLDA backend.

Pipeline:
  1. Load test MFCC segments from test.csv
  2. Extract 192-dim ECAPA embeddings (no classifier head)
  3. Average segment embeddings per utterance (embedding-level fusion)
  4. Load trained PLDA model (plda_model.pkl from Stage 6)
  5. Classify with PLDA log-posterior scores
"""
import json
import os
import re
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

import hyperpyyaml

from utils.extract_config import get_config
from utils.training.plda_trainer import PLDAModel

logger = logging.getLogger(__name__)

EMOTION_LABELS = {
    0: "Happiness", 1: "Neutral", 2: "Anger",
    3: "Fear",      4: "Boredom", 5: "Disgust", 6: "Sadness",
}

_HPARAMS = str(Path(__file__).parent.parent.parent / 'config' / 'ecapa_hparams.yaml')


def _load_ecapa_embedding_model(model_dir, hparams_file=None, device='cpu'):
    """Load embedding_model + normalizer from model.pt (no classifier)."""
    with open(hparams_file or _HPARAMS) as f:
        hparams = hyperpyyaml.load_hyperpyyaml(f)

    payload = torch.load(Path(model_dir) / 'model.pt', map_location='cpu')
    hparams['embedding_model'].load_state_dict(payload['embedding_model'])
    hparams['mean_var_norm'].load_state_dict(payload['normalizer'], strict=False)

    emb_model  = hparams['embedding_model'].eval().to(device)
    normalizer = hparams['mean_var_norm'].to(device)
    return emb_model, normalizer


def _extract_embedding(mfcc_path, emb_model, normalizer, device):
    """Return a 192-dim embedding for one MFCC segment."""
    feat = np.squeeze(np.load(mfcc_path))
    if feat.ndim == 2 and feat.shape[0] <= 256 and feat.shape[1] > feat.shape[0]:
        feat = feat.T
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    t = torch.clamp(
        torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device),
        -50.0, 50.0,
    )
    lens = torch.tensor([1.0], device=device)

    with torch.no_grad():
        t_norm = normalizer(t, lens)
        emb = emb_model(t_norm, lens)   # (1, 192)

    return emb.squeeze().cpu().numpy()  # (192,)


def _score_with_plda(test_csv, emb_model, normalizer, plda_model, device):
    """Extract embeddings, average per utterance, classify with PLDA.

    Returns: y_true (N,), y_pred (N,), ids (N,), probs (N, 7)
    """
    df = pd.read_csv(test_csv)

    utts = defaultdict(lambda: {'paths': [], 'label': None})
    for _, row in df.iterrows():
        seg_id  = str(row.get('ID', row.get('id', '')))
        base_id = re.sub(r'_\d+$', '', seg_id)
        utts[base_id]['paths'].append(str(row['mfcc']))
        utts[base_id]['label'] = int(row['label'])

    y_true, y_pred, ids, all_probs = [], [], [], []

    for utt_id in sorted(utts):
        paths = utts[utt_id]['paths']
        label = utts[utt_id]['label']

        embeddings = []
        for p in paths:
            if not os.path.exists(p):
                logger.warning(f"Missing MFCC: {p}")
                continue
            try:
                embeddings.append(_extract_embedding(p, emb_model, normalizer, device))
            except Exception as e:
                logger.warning(f"Embedding failed for {p}: {e}")

        if not embeddings:
            logger.warning(f"No valid segments for utterance {utt_id}, skipping")
            continue

        # Average segment embeddings → utterance-level embedding
        # Cast to float64: PLDA library requires float64 inputs
        utt_emb = np.mean(embeddings, axis=0, keepdims=True).astype(np.float64)  # (1, 192)

        probs = plda_model.predict_proba(utt_emb)[0]          # (7,)
        # Map PLDA class indices back to emotion label ints
        pred_idx = int(np.argmax(probs))
        pred_label = plda_model.classes[pred_idx]

        # Build full 7-class probability vector aligned to EMOTION_LABELS
        full_probs = np.zeros(len(EMOTION_LABELS))
        for i, cls in enumerate(plda_model.classes):
            full_probs[cls] = probs[i]

        y_true.append(label)
        y_pred.append(pred_label)
        ids.append(utt_id)
        all_probs.append(full_probs)

    return np.array(y_true), np.array(y_pred), ids, np.array(all_probs)


def _plot(results_dir, conf_mat, report):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if conf_mat is not None:
            labels = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(conf_mat, cmap='Blues')
            ax.set_xticks(range(7)); ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(range(7)); ax.set_yticklabels(labels)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, conf_mat[i, j], ha='center', va='center',
                            color='white' if conf_mat[i, j] > conf_mat.max() * 0.5 else 'black',
                            fontsize=8)
            fig.tight_layout()
            fig.savefig(results_dir / 'confusion_matrix.png', dpi=150)
            plt.close(fig)

        if report:
            names = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)
                     if EMOTION_LABELS[i] in report]
            f1s = [report[n]['f1-score'] for n in names]
            if f1s:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(names, f1s, color='#4C78A8')
                ax.set_title('F1 Score per Emotion')
                ax.set_ylabel('F1')
                ax.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(results_dir / 'f1_per_class.png', dpi=150)
                plt.close(fig)
    except Exception:
        pass


def score_speaker_plda(speaker_id, base_dir=None, results_dir=None):
    """Score test speaker: ECAPA embeddings → PLDA classification.

    Requires Stage 4 (--train), Stage 5 (--embeddings), and Stage 6 (--train-plda) to have completed.
    Returns a dict with 'accuracy', 'f1_macro', 'f1_weighted', 'scores_csv'.
    """
    config   = get_config()
    base_dir = Path(base_dir or config['BASE_DIR'])

    model_dir  = base_dir / 'output' / 'models' / f"speaker_{speaker_id}"
    plda_path  = base_dir / 'output' / 'models' / 'plda' / f"speaker_{speaker_id}" / 'plda_model.pkl'
    test_csv   = base_dir / config['PATHS']['LOSO'] / f"speaker_{speaker_id}" / 'test.csv'
    out_dir    = Path(results_dir or (base_dir / config['PATHS'].get('RESULTS', 'output/results'))) \
                 / f"speaker_{speaker_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / 'model.pt').exists():
        raise FileNotFoundError(f"ECAPA model not found: {model_dir / 'model.pt'}")
    if not plda_path.exists():
        raise FileNotFoundError(
            f"NMC model not found: {plda_path}\n"
            "Run --embeddings then --train-plda first."
        )
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Speaker {speaker_id}: ECAPA+PLDA inference on {device}")

    emb_model, normalizer = _load_ecapa_embedding_model(str(model_dir), device=device)
    plda_model = PLDAModel.load(str(plda_path))

    y_true, y_pred, ids, all_probs = _score_with_plda(
        str(test_csv), emb_model, normalizer, plda_model, device
    )

    if len(y_true) == 0:
        raise RuntimeError(f"No utterances scored for speaker {speaker_id}")

    accuracy    = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_micro    = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat    = confusion_matrix(y_true, y_pred, labels=sorted(EMOTION_LABELS))
    report      = classification_report(
        y_true, y_pred,
        labels=sorted(EMOTION_LABELS),
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)],
        zero_division=0, output_dict=True,
    )

    rows = []
    for i, (sid, gt, pred) in enumerate(zip(ids, y_true, y_pred)):
        prob_row = {EMOTION_LABELS[j]: float(f"{all_probs[i, j]:.5f}")
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
    config   = get_config()
    base_dir = Path(base_dir or config['BASE_DIR'])
    models_root = base_dir / 'output' / 'models'

    speakers = sorted(
        d.replace('speaker_', '') for d in os.listdir(models_root)
        if d.startswith('speaker_') and (models_root / d / 'model.pt').exists()
    ) if models_root.exists() else []

    all_results = {}
    for spk in speakers:
        try:
            all_results[spk] = score_speaker_plda(
                spk, base_dir=str(base_dir), results_dir=results_dir
            )
        except Exception as e:
            logger.error(f"Scoring failed for speaker {spk}: {e}")
            all_results[spk] = {'error': str(e)}

    out_root = Path(results_dir or (base_dir / config.get('PATHS', {}).get('RESULTS', 'output/results')))
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [{'speaker': spk, 'accuracy': r.get('accuracy'), 'f1_macro': r.get('f1_macro'),
             'f1_weighted': r.get('f1_weighted'), 'error': r.get('error')}
            for spk, r in all_results.items()]
    pd.DataFrame(rows).to_csv(out_root / 'plda_summary.csv', index=False)

    return all_results
