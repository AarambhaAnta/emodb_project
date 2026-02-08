"""
PLDA scoring utilities using averaged test embeddings and emotion centroids.

This module scores each test embedding against 7 emotion centroids using
SpeechBrain PLDA fast scoring and produces metrics + plots.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from speechbrain.processing.PLDA_LDA import StatObject_SB, Ndx, fast_PLDA_scoring

from utils.extract_config import get_config
from utils.testing.emotion_testing import EMOTION_LABELS, _length_normalize


def _load_plda_model(model_path):
    import pickle
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    if "plda" not in data:
        raise ValueError(f"Invalid PLDA model payload: {model_path}")

    return data


def _apply_whitening(embeddings, mean, mat):
    centered = embeddings - mean.reshape(1, -1)
    return centered @ mat


def _build_stat_object(embeddings, ids):
    modelset = np.array([str(i) for i in ids], dtype="|O")
    segset = np.array([str(i) for i in ids], dtype="|O")
    start = np.array([None] * len(ids))
    stop = np.array([None] * len(ids))
    stat0 = np.ones((len(ids), 1), dtype=np.float64)
    stat1 = np.asarray(embeddings, dtype=np.float64)

    return StatObject_SB(
        modelset=modelset,
        segset=segset,
        start=start,
        stop=stop,
        stat0=stat0,
        stat1=stat1,
    )


def _emotion_id_from_path(path_value):
    match = re.search(r"emotion_(\d+)\.npy$", str(path_value))
    if not match:
        return None
    return int(match.group(1))


def _load_test_averaged_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = {"id", "embedding_path", "emotion_path", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    grouped = {}
    for _, row in df.iterrows():
        sample_id = str(row["id"])
        grouped.setdefault(sample_id, []).append(row)

    test_ids = []
    test_paths = []
    gt_labels = []

    for sample_id, rows in sorted(grouped.items()):
        emb_path = rows[0]["embedding_path"]
        if not os.path.exists(emb_path):
            continue
        test_ids.append(sample_id)
        test_paths.append(emb_path)

        gt = None
        for row in rows:
            try:
                gt_value = int(row["ground_truth"])
            except Exception:
                gt_value = None
            if gt_value == 1:
                gt = _emotion_id_from_path(row["emotion_path"])
                break
        gt_labels.append(gt)

    return test_ids, test_paths, gt_labels


def score_speaker_plda(
    speaker_id,
    base_dir=None,
    model_path=None,
    results_dir=None,
    testing_dir=None
):
    config = get_config()
    base_dir = Path(base_dir or config.get("BASE_DIR", Path(__file__).resolve().parents[2]))
    paths = config.get("PATHS", {})

    if testing_dir is None:
        testing_dir = base_dir / paths.get("TESTING", "data/testing")
    else:
        testing_dir = Path(testing_dir)

    if results_dir is None:
        results_dir = base_dir / paths.get("RESULTS", "output/results")
    else:
        results_dir = Path(results_dir)

    if model_path is None:
        models_root = base_dir / paths.get("MODELS", "output/models")
        model_path = models_root / "plda" / f"speaker_{speaker_id}" / "plda_model.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"PLDA model not found: {model_path}")

    model_payload = _load_plda_model(str(model_path))
    plda = model_payload["plda"]
    whiten_mean = model_payload.get("whiten_mean")
    whiten_mat = model_payload.get("whiten_mat")
    length_norm = model_payload.get("length_norm", True)

    speaker_test_dir = testing_dir / f"speaker_{speaker_id}"
    test_csv = speaker_test_dir / "test_averaged_embeddings.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test averaged CSV not found: {test_csv}")

    test_ids, test_paths, gt_labels = _load_test_averaged_csv(str(test_csv))
    if not test_ids:
        raise ValueError(f"No test embeddings found for speaker {speaker_id}")

    test_embeddings = []
    for path in test_paths:
        emb = np.load(path)
        emb = np.asarray(emb, dtype=np.float64)
        if np.isnan(emb).any() or np.isinf(emb).any():
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        test_embeddings.append(emb)
    test_embeddings = np.vstack(test_embeddings)

    if length_norm:
        test_embeddings = _length_normalize(test_embeddings)
    if whiten_mean is not None and whiten_mat is not None:
        test_embeddings = _apply_whitening(test_embeddings, whiten_mean, whiten_mat)

    # Load emotion centroids for enrollment
    centroids = []
    centroid_ids = []
    for emotion_id in sorted(EMOTION_LABELS.keys()):
        centroid_path = speaker_test_dir / "train" / f"emotion_{emotion_id}.npy"
        if not centroid_path.exists():
            continue
        emb = np.load(centroid_path)
        emb = np.asarray(emb, dtype=np.float64)
        if np.isnan(emb).any() or np.isinf(emb).any():
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        centroids.append(emb)
        centroid_ids.append(emotion_id)

    if not centroids:
        raise ValueError(f"No centroids found for speaker {speaker_id}")

    centroids = np.vstack(centroids)
    if length_norm:
        centroids = _length_normalize(centroids)
    if whiten_mean is not None and whiten_mat is not None:
        centroids = _apply_whitening(centroids, whiten_mean, whiten_mat)

    enroll_stat = _build_stat_object(centroids, centroid_ids)
    test_stat = _build_stat_object(test_embeddings, test_ids)

    model_ids = np.array([str(e) for e in centroid_ids], dtype="|O")
    seg_ids = np.array([str(t) for t in test_ids], dtype="|O")
    models = np.repeat(model_ids, len(seg_ids))
    testsegs = np.tile(seg_ids, len(model_ids))
    ndx = Ndx(models=models, testsegs=testsegs)

    scores = fast_PLDA_scoring(
        enroll=enroll_stat,
        test=test_stat,
        ndx=ndx,
        mu=plda.mean,
        F=plda.F,
        Sigma=plda.Sigma,
        p_known=0.0,
        scaling_factor=1.0,
        check_missing=True,
    )

    scoremat = scores.scoremat
    pred_idx = np.argmax(scoremat, axis=0)
    pred_labels = [int(model_ids[i]) for i in pred_idx]
    pred_scores = scoremat[pred_idx, np.arange(scoremat.shape[1])]

    results_dir = results_dir / f"speaker_{speaker_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, test_id in enumerate(test_ids):
        row = {
            "id": test_id,
            "ground_truth": gt_labels[idx],
            "prediction": pred_labels[idx],
            "score": float(pred_scores[idx])
        }
        for e_idx, emotion_id in enumerate(model_ids):
            row[f"score_emotion_{emotion_id}"] = float(scoremat[e_idx, idx])
        rows.append(row)

    scores_csv = results_dir / "plda_scores.csv"
    pd.DataFrame(rows).to_csv(scores_csv, index=False)

    # Metrics
    valid_indices = [i for i, gt in enumerate(gt_labels) if gt is not None]
    if valid_indices:
        y_true = [gt_labels[i] for i in valid_indices]
        y_pred = [pred_labels[i] for i in valid_indices]
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(
            y_true,
            y_pred,
            labels=sorted(EMOTION_LABELS.keys()),
            target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())],
            zero_division=0,
            output_dict=True,
        )
        conf_mat = confusion_matrix(
            y_true,
            y_pred,
            labels=sorted(EMOTION_LABELS.keys())
        )
    else:
        accuracy = f1_macro = f1_micro = f1_weighted = None
        report = {}
        conf_mat = None

    metrics = {
        "speaker": speaker_id,
        "model_path": str(model_path),
        "test_csv": str(test_csv),
        "scores_csv": str(scores_csv),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist() if conf_mat is not None else None,
    }

    metrics_json = results_dir / "plda_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_metrics(
        results_dir=results_dir,
        confusion_matrix=conf_mat,
        report=report
    )

    return metrics


def _plot_metrics(results_dir, confusion_matrix, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if confusion_matrix is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(confusion_matrix, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(EMOTION_LABELS)))
        ax.set_yticks(range(len(EMOTION_LABELS)))
        ax.set_xticklabels([EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())], rotation=45, ha="right")
        ax.set_yticklabels([EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())])
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    if report:
        f1_scores = []
        labels = []
        for label_id in sorted(EMOTION_LABELS.keys()):
            label_name = EMOTION_LABELS[label_id]
            if label_name in report:
                f1_scores.append(report[label_name]["f1-score"])
                labels.append(label_name)

        if f1_scores:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, f1_scores, color="#4C78A8")
            ax.set_title("F1 Score per Emotion")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(results_dir / "f1_per_class.png", dpi=150)
            plt.close(fig)


def score_all_speakers_plda(base_dir=None, model_root=None, results_dir=None, testing_dir=None):
    config = get_config()
    base_dir = Path(base_dir or config.get("BASE_DIR", Path(__file__).resolve().parents[2]))
    paths = config.get("PATHS", {})

    if testing_dir is None:
        testing_dir = base_dir / paths.get("TESTING", "data/testing")
    else:
        testing_dir = Path(testing_dir)

    speaker_dirs = sorted([d.name for d in testing_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    results = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            model_path = None
            if model_root is not None:
                model_path = Path(model_root) / f"speaker_{speaker_id}" / "plda_model.pkl"
            results[speaker_id] = score_speaker_plda(
                speaker_id=speaker_id,
                base_dir=base_dir,
                model_path=model_path,
                results_dir=results_dir,
                testing_dir=testing_dir,
            )
        except Exception as exc:
            results[speaker_id] = {"error": str(exc)}

    return results
