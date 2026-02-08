#!/usr/bin/env python3
"""
Test emotion classification using speaker-level emotion centroids.

This script averages train embeddings per emotion (7 centroids per speaker)
then classifies test embeddings by cosine similarity.

Usage:
  python test_emotion_models.py --speaker 03
  python test_emotion_models.py --all
"""
import argparse
from pathlib import Path

from utils.testing import test_all_speakers, test_speaker_emotions


def main():
    parser = argparse.ArgumentParser(
        description="Test emotion classification with centroid embeddings"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--speaker", type=str, help="Speaker ID (e.g., 03)")
    mode_group.add_argument("--all", action="store_true", help="Test all speakers")

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Project base directory (default: repo root)"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Embeddings directory (default: data/embeddings)"
    )
    parser.add_argument(
        "--train-out-dir",
        type=str,
        default=None,
        help="Output dir for centroids (default: data/training)"
    )
    parser.add_argument(
        "--test-out-dir",
        type=str,
        default=None,
        help="Output dir for test results (default: data/testing)"
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else None

    if args.speaker:
        result = test_speaker_emotions(
            speaker_id=args.speaker,
            base_dir=base_dir,
            embeddings_dir=args.embeddings_dir,
            train_out_dir=args.train_out_dir,
            test_out_dir=args.test_out_dir
        )
        print(f"Speaker {result.speaker_id} accuracy: {result.accuracy:.4f}")
        print(f"Results saved in: {result.output_dir}")
    else:
        results = test_all_speakers(
            base_dir=base_dir,
            embeddings_dir=args.embeddings_dir,
            train_out_dir=args.train_out_dir,
            test_out_dir=args.test_out_dir
        )
        print("Testing complete. Summary:")
        for speaker_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"  speaker {speaker_id}: ERROR {result['error']}")
            else:
                print(f"  speaker {speaker_id}: accuracy {result.accuracy:.4f}")


if __name__ == "__main__":
    main()
