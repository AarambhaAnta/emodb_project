"""
Emotion testing utilities using speaker-level emotion centroids.

For each speaker, this module averages train embeddings per emotion to produce
7 centroid embeddings, then classifies test embeddings by cosine similarity.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.extract_config import get_config


EMOTION_LABELS = {
    0: "Happiness",
    1: "Neutral",
    2: "Anger",
    3: "Fear",
    4: "Boredom",
    5: "Disgust",
    6: "Sadness"
}


def _length_normalize(embeddings, eps=1e-12):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms


def _resolve_path(base_dir, path_value):
    if pd.isna(path_value):
        return None
    path_str = str(path_value)
    if os.path.isabs(path_str):
        return path_str
    return str(Path(base_dir) / path_str)


def _load_embeddings_from_csv(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    if "embedding" not in df.columns:
        raise ValueError(f"Missing 'embedding' column in {csv_path}")

    embeddings = []
    labels = []
    ids = []
    for _, row in df.iterrows():
        emb_path = _resolve_path(base_dir, row["embedding"])
        if not emb_path or not os.path.exists(emb_path):
            continue
        emb = np.load(emb_path)
        emb = np.asarray(emb, dtype=np.float64)
        if np.isnan(emb).any() or np.isinf(emb).any():
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings.append(emb)
        labels.append(int(row["label"]))
        ids.append(row.get("id", Path(emb_path).stem))

    if not embeddings:
        raise ValueError(f"No embeddings loaded from {csv_path}")

    return np.vstack(embeddings), np.array(labels), ids


def _base_id(sample_id):
    if sample_id is None:
        return None
    text = str(sample_id)
    if "_" not in text:
        return text
    prefix, suffix = text.rsplit("_", 1)
    if suffix.isdigit():
        return prefix
    return text


def _is_first_part_id(sample_id):
    if sample_id is None:
        return False
    text = str(sample_id)
    if "_" not in text:
        return True
    _, suffix = text.rsplit("_", 1)
    if suffix.isdigit():
        return int(suffix) == 1
    return True


def _filter_first_part_arrays(embeddings, labels, ids):
    keep_mask = np.array([_is_first_part_id(sample_id) for sample_id in ids], dtype=bool)
    if not np.any(keep_mask):
        raise ValueError("No first-part samples found")

    filtered_embeddings = embeddings[keep_mask]
    filtered_labels = labels[keep_mask]
    filtered_ids = [sample_id for sample_id, keep in zip(ids, keep_mask) if keep]

    return filtered_embeddings, filtered_labels, filtered_ids


@dataclass
class EmotionCentroidResult:
    speaker_id: str
    train_count: int
    test_count: int
    accuracy: float
    output_dir: str


class EmotionCentroidTester:
    """Compute emotion centroids and classify test embeddings by cosine similarity."""

    def __init__(
        self,
        base_dir=None,
        embeddings_dir=None,
        train_out_dir=None,
        test_out_dir=None,
        train_csv_name="train_embeddings.csv",
        test_csv_name="test_embeddings.csv",
    ):
        config = get_config()
        self.base_dir = Path(base_dir or config.get("BASE_DIR", Path(__file__).resolve().parents[2]))

        paths = config.get("PATHS", {})
        embeddings_default = self.base_dir / paths.get("EMBEDDINGS", "data/embeddings")
        training_default = self.base_dir / paths.get("TRAINING", "data/training")
        testing_default = self.base_dir / paths.get("TESTING", "data/testing")

        self.embeddings_dir = Path(embeddings_dir or embeddings_default)
        self.train_out_dir = Path(train_out_dir or training_default)
        self.test_out_dir = Path(test_out_dir or testing_default)
        self.train_csv_name = train_csv_name
        self.test_csv_name = test_csv_name

    def average_train_embeddings(self, speaker_id, output_base=None):
        """Average train embeddings per emotion and save to data/testing/speaker_{id}/train."""
        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        train_csv = speaker_dir / self.train_csv_name
        if not train_csv.exists():
            raise FileNotFoundError(f"Training embeddings CSV not found: {train_csv}")

        X_train, y_train, _ = _load_embeddings_from_csv(str(train_csv), self.base_dir)
        X_train = _length_normalize(X_train)

        output_root = Path(output_base) if output_base else self.test_out_dir
        out_dir = output_root / f"speaker_{speaker_id}" / "train"
        out_dir.mkdir(parents=True, exist_ok=True)

        centroid_paths = []
        centroid_labels = []
        centroid_counts = []

        for label in sorted(EMOTION_LABELS.keys()):
            mask = y_train == label
            if not np.any(mask):
                continue
            centroid = np.mean(X_train[mask], axis=0)
            centroid = _length_normalize(centroid.reshape(1, -1)).squeeze(0)
            out_path = out_dir / f"emotion_{label}.npy"
            np.save(out_path, centroid)
            centroid_paths.append(str(out_path))
            centroid_labels.append(label)
            centroid_counts.append(int(np.sum(mask)))

        if not centroid_paths:
            raise ValueError(f"No centroids computed for speaker {speaker_id}")

        summary_df = pd.DataFrame({
            "emotion_id": centroid_labels,
            "emotion": [EMOTION_LABELS[l] for l in centroid_labels],
            "n_samples": centroid_counts,
            "embedding": centroid_paths
        })
        summary_csv = output_root / f"speaker_{speaker_id}" / "emotion_centroids.csv"
        summary_df.to_csv(summary_csv, index=False)

        return summary_csv

    def build_test_embeddings_noavg(self, speaker_id, output_base=None):
        """Build test CSV without averaging parts; one row per segment per emotion."""
        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        test_csv = speaker_dir / self.test_csv_name
        if not test_csv.exists():
            raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

        df = pd.read_csv(test_csv)
        if "embedding" not in df.columns:
            raise ValueError(f"Missing 'embedding' column in {test_csv}")

        output_root = Path(output_base) if output_base else self.test_out_dir
        out_dir = output_root / f"speaker_{speaker_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for _, row in df.iterrows():
            sample_id = row.get("id")
            label = row.get("label")
            emb_path = _resolve_path(self.base_dir, row["embedding"])
            if not sample_id or emb_path is None:
                continue
            for emotion_id in sorted(EMOTION_LABELS.keys()):
                emotion_path = self.train_out_dir / f"speaker_{speaker_id}" / "train" / f"emotion_{emotion_id}.npy"
                if pd.isna(label):
                    gt_value = None
                else:
                    gt_value = 1 if int(emotion_id) == int(label) else 0
                rows.append({
                    "id": str(sample_id),
                    "embedding_path": str(emb_path),
                    "emotion_path": str(emotion_path),
                    "ground_truth": gt_value
                })

        if not rows:
            raise ValueError(f"No rows created from {test_csv}")

        summary_csv = out_dir / "test_noavg_embeddings.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

        return summary_csv

    def average_test_embeddings(self, speaker_id, output_base=None):
        """Average test embeddings by base id and save to data/testing/speaker_{id}/test."""
        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        test_csv = speaker_dir / self.test_csv_name
        if not test_csv.exists():
            raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

        X_test, y_test, ids = _load_embeddings_from_csv(str(test_csv), self.base_dir)

        output_root = Path(output_base) if output_base else self.test_out_dir
        out_dir = output_root / f"speaker_{speaker_id}" / "test"
        out_dir.mkdir(parents=True, exist_ok=True)

        grouped = {}
        label_groups = {}
        for emb, sample_id, label in zip(X_test, ids, y_test):
            base = _base_id(sample_id)
            if base is None:
                continue
            grouped.setdefault(base, []).append(emb)
            label_groups.setdefault(base, []).append(int(label))

        if not grouped:
            raise ValueError(f"No test embeddings grouped for speaker {speaker_id}")

        rows = []
        for base, emb_list in sorted(grouped.items()):
            labels = label_groups.get(base, [])
            if labels:
                values, counts = np.unique(labels, return_counts=True)
                ground_truth = int(values[np.argmax(counts)])
            else:
                ground_truth = None
            avg_emb = np.mean(np.vstack(emb_list), axis=0)
            out_path = out_dir / f"{base}.npy"
            np.save(out_path, avg_emb)
            for emotion_id in sorted(EMOTION_LABELS.keys()):
                emotion_path = self.train_out_dir / f"speaker_{speaker_id}" / "train" / f"emotion_{emotion_id}.npy"
                if ground_truth is None:
                    gt_value = None
                else:
                    gt_value = 1 if emotion_id == ground_truth else 0
                rows.append({
                    "id": base,
                    "embedding_path": str(out_path),
                    "emotion_path": str(emotion_path),
                    "ground_truth": gt_value
                })

        summary_csv = output_root / f"speaker_{speaker_id}" / "test_averaged_embeddings.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

        return summary_csv

    def build_test_embeddings_firstpart(self, speaker_id, output_base=None):
        """Pick the first segment per base id and save to data/testing/speaker_{id}/test."""
        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        test_csv = speaker_dir / self.test_csv_name
        if not test_csv.exists():
            raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

        df = pd.read_csv(test_csv)
        if "embedding" not in df.columns:
            raise ValueError(f"Missing 'embedding' column in {test_csv}")

        output_root = Path(output_base) if output_base else self.test_out_dir
        out_dir = output_root / f"speaker_{speaker_id}" / "test"
        out_dir.mkdir(parents=True, exist_ok=True)

        grouped = {}
        for _, row in df.iterrows():
            sample_id = row.get("id")
            label = row.get("label")
            emb_path = _resolve_path(self.base_dir, row["embedding"])
            if not sample_id or emb_path is None or not os.path.exists(emb_path):
                continue
            base = _base_id(sample_id)
            if base is None:
                continue
            grouped.setdefault(base, []).append((str(sample_id), emb_path, label))

        if not grouped:
            raise ValueError(f"No test embeddings grouped for speaker {speaker_id}")

        rows = []
        for base, items in sorted(grouped.items()):
            items_sorted = sorted(items, key=lambda entry: entry[0])
            sample_id, emb_path, label = items_sorted[0]

            emb = np.load(emb_path)
            emb = np.asarray(emb, dtype=np.float64)
            if np.isnan(emb).any() or np.isinf(emb).any():
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

            out_path = out_dir / f"{base}.npy"
            np.save(out_path, emb)

            for emotion_id in sorted(EMOTION_LABELS.keys()):
                emotion_path = self.train_out_dir / f"speaker_{speaker_id}" / "train" / f"emotion_{emotion_id}.npy"
                if pd.isna(label):
                    gt_value = None
                else:
                    gt_value = 1 if int(emotion_id) == int(label) else 0
                rows.append({
                    "id": base,
                    "embedding_path": str(out_path),
                    "emotion_path": str(emotion_path),
                    "ground_truth": gt_value
                })

        summary_csv = output_root / f"speaker_{speaker_id}" / "test_first_embeddings.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

        return summary_csv

    def compute_centroids(self, speaker_id):
        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        train_csv = speaker_dir / self.train_csv_name
        if not train_csv.exists():
            raise FileNotFoundError(f"Training embeddings CSV not found: {train_csv}")

        X_train, y_train, _ = _load_embeddings_from_csv(str(train_csv), self.base_dir)
        X_train = _length_normalize(X_train)

        train_count = int(len(y_train))

        centroids = []
        centroid_labels = []
        centroid_counts = []

        for label in sorted(EMOTION_LABELS.keys()):
            mask = y_train == label
            if not np.any(mask):
                continue
            centroid = np.mean(X_train[mask], axis=0)
            centroids.append(centroid)
            centroid_labels.append(label)
            centroid_counts.append(int(np.sum(mask)))

        if not centroids:
            raise ValueError(f"No centroids computed for speaker {speaker_id}")

        centroids = np.vstack(centroids)
        centroids = _length_normalize(centroids)

        out_dir = self.train_out_dir / f"speaker_{speaker_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        centroid_paths = []
        for label, centroid in zip(centroid_labels, centroids):
            out_path = out_dir / f"emotion_{label}.npy"
            np.save(out_path, centroid)
            centroid_paths.append(str(out_path))

        summary_df = pd.DataFrame({
            "emotion_id": centroid_labels,
            "emotion": [EMOTION_LABELS[l] for l in centroid_labels],
            "n_samples": centroid_counts,
            "embedding": centroid_paths
        })
        summary_csv = out_dir / "emotion_centroids.csv"
        summary_df.to_csv(summary_csv, index=False)

        return centroids, centroid_labels, summary_csv, train_count

    def evaluate(self, speaker_id):
        centroids, centroid_labels, _, train_count = self.compute_centroids(speaker_id)

        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        test_csv = speaker_dir / self.test_csv_name
        if not test_csv.exists():
            raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

        X_test, y_test, ids = _load_embeddings_from_csv(str(test_csv), self.base_dir)
        X_test = _length_normalize(X_test)

        scores = X_test @ centroids.T
        pred_idx = np.argmax(scores, axis=1)
        y_pred = np.array([centroid_labels[i] for i in pred_idx])

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            labels=sorted(EMOTION_LABELS.keys()),
            target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())],
            zero_division=0,
            output_dict=True
        )
        conf_mat = confusion_matrix(y_test, y_pred, labels=sorted(EMOTION_LABELS.keys()))

        out_dir = self.test_out_dir / f"speaker_{speaker_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        preds_df = pd.DataFrame({
            "id": ids,
            "label": y_test,
            "label_name": [EMOTION_LABELS[i] for i in y_test],
            "pred": y_pred,
            "pred_name": [EMOTION_LABELS[i] for i in y_pred],
            "score": scores[np.arange(len(scores)), pred_idx]
        })
        preds_csv = out_dir / "test_predictions.csv"
        preds_df.to_csv(preds_csv, index=False)

        results = {
            "speaker": speaker_id,
            "train_embeddings_csv": str((speaker_dir / "train_embeddings.csv")),
            "test_embeddings_csv": str(test_csv),
            "train_centroids_csv": str((self.train_out_dir / f"speaker_{speaker_id}" / "emotion_centroids.csv")),
            "test_predictions_csv": str(preds_csv),
            "accuracy": float(accuracy),
            "n_train_samples": int(train_count),
            "n_test_samples": int(len(y_test)),
            "confusion_matrix": conf_mat.tolist(),
            "classification_report": report
        }

        results_path = out_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return EmotionCentroidResult(
            speaker_id=speaker_id,
            train_count=int(train_count),
            test_count=int(len(y_test)),
            accuracy=float(accuracy),
            output_dir=str(out_dir)
        )

    def evaluate_firstpart(self, speaker_id):
        centroids, centroid_labels, _, train_count = self.compute_centroids(speaker_id)

        speaker_dir = self.embeddings_dir / f"speaker_{speaker_id}"
        test_csv = speaker_dir / self.test_csv_name
        if not test_csv.exists():
            raise FileNotFoundError(f"Test embeddings CSV not found: {test_csv}")

        X_test, y_test, ids = _load_embeddings_from_csv(str(test_csv), self.base_dir)
        X_test, y_test, ids = _filter_first_part_arrays(X_test, y_test, ids)
        X_test = _length_normalize(X_test)

        scores = X_test @ centroids.T
        pred_idx = np.argmax(scores, axis=1)
        y_pred = np.array([centroid_labels[i] for i in pred_idx])

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            labels=sorted(EMOTION_LABELS.keys()),
            target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())],
            zero_division=0,
            output_dict=True
        )
        conf_mat = confusion_matrix(y_test, y_pred, labels=sorted(EMOTION_LABELS.keys()))

        out_dir = self.test_out_dir / f"speaker_{speaker_id}" / "firstpart"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "speaker_id": speaker_id,
            "train_count": int(train_count),
            "test_count": int(len(y_test)),
            "accuracy": float(accuracy),
            "confusion_matrix": conf_mat.tolist(),
            "classification_report": report,
            "first_part_only": True
        }

        results_path = out_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return EmotionCentroidResult(
            speaker_id=speaker_id,
            train_count=int(train_count),
            test_count=int(len(y_test)),
            accuracy=float(accuracy),
            output_dir=str(out_dir)
        )


def test_speaker_emotions(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    train_out_dir=None,
    test_out_dir=None
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_out_dir=train_out_dir,
        test_out_dir=test_out_dir
    )
    return tester.evaluate(speaker_id)


def test_speaker_emotions_firstpart(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    train_out_dir=None,
    test_out_dir=None
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_out_dir=train_out_dir,
        test_out_dir=test_out_dir
    )
    return tester.evaluate_firstpart(speaker_id)


def test_all_speakers(
    base_dir=None,
    embeddings_dir=None,
    train_out_dir=None,
    test_out_dir=None
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_out_dir=train_out_dir,
        test_out_dir=test_out_dir
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    results = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            results[speaker_id] = tester.evaluate(speaker_id)
        except Exception as exc:
            results[speaker_id] = {"error": str(exc)}

    return results


def test_all_speakers_firstpart(
    base_dir=None,
    embeddings_dir=None,
    train_out_dir=None,
    test_out_dir=None
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_out_dir=train_out_dir,
        test_out_dir=test_out_dir
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    results = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            results[speaker_id] = tester.evaluate_firstpart(speaker_id)
        except Exception as exc:
            results[speaker_id] = {"error": str(exc)}

    return results


def average_emotion_embeddings_for_speaker(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    train_csv_name="train_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_csv_name=train_csv_name
    )
    return tester.average_train_embeddings(speaker_id, output_base=output_base)


def average_emotion_embeddings_for_all(
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    train_csv_name="train_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        train_csv_name=train_csv_name
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    outputs = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            outputs[speaker_id] = tester.average_train_embeddings(speaker_id, output_base=output_base)
        except Exception as exc:
            outputs[speaker_id] = {"error": str(exc)}

    return outputs


def average_test_embeddings_for_speaker(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )
    return tester.average_test_embeddings(speaker_id, output_base=output_base)


def average_test_embeddings_for_all(
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    outputs = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            outputs[speaker_id] = tester.average_test_embeddings(speaker_id, output_base=output_base)
        except Exception as exc:
            outputs[speaker_id] = {"error": str(exc)}

    return outputs


def build_test_noavg_for_speaker(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )
    return tester.build_test_embeddings_noavg(speaker_id, output_base=output_base)


def build_test_firstpart_for_speaker(
    speaker_id,
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )
    return tester.build_test_embeddings_firstpart(speaker_id, output_base=output_base)


def build_test_noavg_for_all(
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    outputs = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            outputs[speaker_id] = tester.build_test_embeddings_noavg(speaker_id, output_base=output_base)
        except Exception as exc:
            outputs[speaker_id] = {"error": str(exc)}

    return outputs


def build_test_firstpart_for_all(
    base_dir=None,
    embeddings_dir=None,
    output_base=None,
    test_csv_name="test_embeddings.csv"
):
    tester = EmotionCentroidTester(
        base_dir=base_dir,
        embeddings_dir=embeddings_dir,
        test_csv_name=test_csv_name
    )

    if not tester.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {tester.embeddings_dir}")

    speaker_dirs = sorted([d.name for d in tester.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")])
    outputs = {}

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.replace("speaker_", "")
        try:
            outputs[speaker_id] = tester.build_test_embeddings_firstpart(speaker_id, output_base=output_base)
        except Exception as exc:
            outputs[speaker_id] = {"error": str(exc)}

    return outputs
