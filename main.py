#!/usr/bin/env python3
"""
EmoDb Emotion Recognition Pipeline - Main Entry Point

7-stage pipeline:
  1. --segment    : Segment audio files and create CSV
  2. --mfcc       : Extract MFCC features (396x40)
  3. --loso       : Create LOSO splits + 80/20 train/val splits
  4. --train      : Train ECAPA-TDNN model per speaker
  5. --embeddings : Extract embeddings from trained models
  6. --train-plda : Train PLDA model per speaker
  7. --test-plda  : Test PLDA model per speaker

Usage:
    python main.py --all
    python main.py --segment
    python main.py --mfcc
    python main.py --loso
    python main.py --train [--speaker 03]
    python main.py --embeddings [--speaker 03]
    python main.py --train-plda [--speaker 03]
    python main.py --test-plda [--speaker 03]
    python main.py --all --config path/to/config.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch

from utils import get_config
from utils.audio_processing import segment_audio
from utils.features_extraction import extract_mfcc, create_splits
from utils.training import train_all_speakers, train_speaker_model


def setup_logging(log_dir=None):
    """Set up logging to file and stdout."""
    if log_dir is None:
        config = get_config()
        log_dir = Path(config['BASE_DIR']) / config['PATHS']['LOGS']
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def segment_stage(config, logger):
    """Stage 1: Segment audio files and create CSV."""
    logger.info("=" * 70)
    logger.info("STAGE 1: AUDIO SEGMENTATION")
    logger.info("=" * 70)

    raw_dir = os.path.join(config['BASE_DIR'], config['PATHS']['RAW_DATA'])
    output_dir = os.path.join(config['BASE_DIR'], config['PATHS']['SEGMENT'])
    csv_path = os.path.join(config['BASE_DIR'], config['PATHS']['CSV'], 'segments.csv')

    logger.info(f"Input directory: {raw_dir}")
    logger.info(f"Segment directory: {output_dir}")
    logger.info(f"Output CSV: {csv_path}")

    df = segment_audio(raw_dir=raw_dir, output_dir=output_dir, csv_path=csv_path)
    logger.info(f"Created {len(df)} segments")
    logger.info("Stage 1 complete!\n")
    return csv_path


def mfcc_stage(config, logger):
    """Stage 2: Extract MFCC features (396x40)."""
    logger.info("=" * 70)
    logger.info("STAGE 2: MFCC FEATURE EXTRACTION")
    logger.info("=" * 70)

    input_csv = os.path.join(config['BASE_DIR'], config['PATHS']['CSV'], 'segments.csv')
    output_dir = os.path.join(config['BASE_DIR'], config['MFCC']['OUTPUT_DIR'])
    csv_path = os.path.join(config['BASE_DIR'], config['PATHS']['CSV'], 'mfcc_features.csv')

    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Features directory: {output_dir}")
    logger.info(f"Output CSV: {csv_path}")

    df = extract_mfcc(input_csv=input_csv, output_dir=output_dir, csv_path=csv_path)
    logger.info(f"Extracted {len(df)} MFCC features")
    logger.info("Stage 2 complete!\n")
    return csv_path


def loso_stage(config, logger):
    """Stage 3: Create LOSO + 80/20 train/val/test splits (CSV only, no file copying)."""
    logger.info("=" * 70)
    logger.info("STAGE 3: LOSO + TRAIN/VAL SPLITS")
    logger.info("=" * 70)

    mfcc_csv = os.path.join(config['BASE_DIR'], config['PATHS']['CSV'], 'mfcc_features.csv')
    output_dir = os.path.join(config['BASE_DIR'], config['PATHS']['LOSO'])

    logger.info(f"MFCC CSV: {mfcc_csv}")
    logger.info(f"LOSO directory: {output_dir}")

    splits_info = create_splits(
        mfcc_csv=mfcc_csv,
        output_dir=output_dir,
    )
    logger.info(f"Created splits for {len(splits_info)} speakers: {sorted(splits_info.keys())}")
    logger.info("Stage 3 complete!\n")
    return list(splits_info.keys())


def train_stage(config, logger, speaker=None):
    """Stage 4: Train ECAPA-TDNN model per speaker."""
    logger.info("=" * 70)
    logger.info("STAGE 4: ECAPA-TDNN TRAINING")
    logger.info("=" * 70)

    loso_dir = Path(config['BASE_DIR']) / config['PATHS']['LOSO']
    output_dir = Path(config['BASE_DIR']) / config['PATHS']['MODELS']
    hparams_file = Path(config['BASE_DIR']) / 'config' / 'ecapa_hparams.yaml'

    logger.info(f"LOSO directory: {loso_dir}")
    logger.info(f"Output directory: {output_dir}")

    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    if torch.cuda.is_available():
        run_opts = {"device": "cuda"}
    elif torch.backends.mps.is_available():
        run_opts = {"device": "mps"}
    else:
        run_opts = {"device": "cpu"}
    logger.info(f"Device: {run_opts['device']}")

    if speaker:
        logger.info(f"Training single speaker: {speaker}")
        best_error = train_speaker_model(
            speaker_id=speaker,
            loso_dir=str(loso_dir),
            output_dir=str(output_dir),
            hparams=hparams,
            run_opts=run_opts
        )
        results = {f"speaker_{speaker}": {"status": "success", "best_error": float(best_error)}}
        results_file = output_dir / f'training_results_speaker_{speaker}.json'
    else:
        logger.info("Training all speakers...")
        results = train_all_speakers(
            loso_dir=str(loso_dir),
            output_dir=str(output_dir),
            hparams_file=hparams,
            run_opts=run_opts
        )
        successful = [s for s, r in results.items() if r['status'] == 'success']
        failed = [s for s, r in results.items() if r['status'] == 'failed']
        logger.info(f"Successful: {len(successful)}/{len(results)}, Failed: {len(failed)}")
        if successful:
            avg_error = sum(results[s]['best_error'] for s in successful) / len(successful)
            logger.info(f"Average validation error: {avg_error:.4f}")
        results_file = output_dir / 'training_results.json'

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    logger.info("Stage 4 complete!\n")


def embeddings_stage(config, logger, speaker=None):
    """Stage 5: Extract embeddings from trained ECAPA-TDNN models."""
    logger.info("=" * 70)
    logger.info("STAGE 5: EMBEDDING EXTRACTION")
    logger.info("=" * 70)

    from utils.features_extraction.extract_embeddings import extract_embeddings_for_speaker

    loso_dir = str(Path(config['BASE_DIR']) / config['PATHS']['LOSO'])
    model_dir = str(Path(config['BASE_DIR']) / config['PATHS']['MODELS'])
    output_dir = str(Path(config['BASE_DIR']) / config['PATHS']['EMBEDDINGS'])
    hparams_file = str(Path(config['BASE_DIR']) / 'config' / 'ecapa_hparams.yaml')

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    if speaker:
        speakers = [speaker]
    else:
        speakers = sorted([
            d.replace('speaker_', '')
            for d in os.listdir(loso_dir)
            if d.startswith('speaker_')
        ])

    logger.info(f"Extracting embeddings for {len(speakers)} speaker(s)")

    for spk in speakers:
        model_path = Path(model_dir) / f"speaker_{spk}" / "model.pt"
        if not model_path.exists():
            logger.warning(f"No trained model for speaker {spk}, skipping.")
            continue
        try:
            extract_embeddings_for_speaker(
                speaker_id=spk,
                loso_dir=loso_dir,
                model_dir=model_dir,
                output_dir=output_dir,
                hparams_file=hparams_file,
                device=device
            )
            logger.info(f"Embeddings extracted for speaker {spk}")
        except Exception as e:
            logger.error(f"Failed embeddings for speaker {spk}: {e}")

    logger.info("Stage 5 complete!\n")


def train_plda_stage(config, logger, speaker=None):
    """Stage 6: Train PLDA model per speaker."""
    logger.info("=" * 70)
    logger.info("STAGE 6: PLDA TRAINING")
    logger.info("=" * 70)

    from utils.training.plda_trainer import train_speaker_plda, train_all_speakers_plda
    import yaml

    loso_dir = str(Path(config['BASE_DIR']) / config['PATHS']['LOSO'])
    output_dir = str(Path(config['BASE_DIR']) / 'output' / 'models' / 'plda')
    embeddings_dir = str(Path(config['BASE_DIR']) / config['PATHS']['EMBEDDINGS'])

    # Read PLDA dimensionality from plda_hparams.yaml
    plda_hparams_path = Path(config['BASE_DIR']) / 'config' / 'plda_hparams.yaml'
    n_components = None
    if plda_hparams_path.exists():
        with open(plda_hparams_path) as f:
            plda_hparams = yaml.safe_load(f)
        n_components = plda_hparams.get('plda_dim', None)
    logger.info(f"PLDA n_components: {n_components}")

    if not os.path.exists(embeddings_dir) or not os.listdir(embeddings_dir):
        logger.error("No embeddings found. Run --embeddings (Stage 5) first.")
        return

    if speaker:
        results = train_speaker_plda(
            speaker_id=speaker,
            loso_dir=loso_dir,
            output_dir=output_dir,
            embeddings_dir=embeddings_dir,
            n_components=n_components,
        )
        logger.info(f"PLDA model saved to: {results['model_path']}")
    else:
        all_results = train_all_speakers_plda(
            loso_dir=loso_dir,
            output_dir=output_dir,
            embeddings_dir=embeddings_dir,
            n_components=n_components,
        )
        successful = [s for s, r in all_results.items() if 'model_path' in r]
        logger.info(f"PLDA trained for {len(successful)}/{len(all_results)} speakers")

    logger.info("Stage 6 complete!\n")


def test_plda_stage(config, logger, speaker=None):
    """Stage 7: Test PLDA model per speaker."""
    logger.info("=" * 70)
    logger.info("STAGE 7: PLDA TESTING")
    logger.info("=" * 70)

    from utils.testing.plda_scoring import score_speaker_plda, score_all_speakers_plda

    base_dir = config['BASE_DIR']
    embeddings_dir = str(Path(base_dir) / config['PATHS']['EMBEDDINGS'])
    results_dir = str(Path(base_dir) / config['PATHS'].get('RESULTS', 'output/results'))

    if speaker:
        try:
            metrics = score_speaker_plda(
                speaker_id=speaker, base_dir=base_dir, results_dir=results_dir
            )
            acc = metrics.get('accuracy')
            if acc is not None:
                logger.info(f"Speaker {speaker} accuracy: {acc:.4f} ({acc*100:.2f}%)")
        except Exception as e:
            logger.error(f"PLDA scoring failed for speaker {speaker}: {e}")
    else:
        if not Path(embeddings_dir).exists():
            logger.error("No embeddings found. Run --embeddings (Stage 5) first.")
            return
        all_metrics = score_all_speakers_plda(base_dir=base_dir, results_dir=results_dir)
        accs = [r['accuracy'] for r in all_metrics.values()
                if isinstance(r, dict) and r.get('accuracy') is not None]
        if accs:
            logger.info(f"Mean accuracy across speakers: {sum(accs)/len(accs):.4f}")
        logger.info(f"Results saved to: {results_dir}")

    logger.info("Stage 7 complete!\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EmoDb Emotion Recognition Pipeline (7 stages)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  1  --segment     Segment audio files and create CSV
  2  --mfcc        Extract MFCC features (396x40)
  3  --loso        Create LOSO + 80/20 train/val splits
  4  --train       Train ECAPA-TDNN model per speaker
  5  --embeddings  Extract embeddings from trained models
  6  --train-plda  Train PLDA model per speaker
  7  --test-plda   Test PLDA model per speaker

Examples:
  python main.py --all                    # Run all 7 stages
  python main.py --segment --mfcc --loso  # Stages 1-3
  python main.py --train --speaker 03     # Stage 4 for one speaker
  python main.py --embeddings --speaker 03
  python main.py --train-plda --speaker 03
  python main.py --test-plda --speaker 03
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (all 7 stages)')
    parser.add_argument('--segment', action='store_true',
                        help='Stage 1: Segment audio files and create CSV')
    parser.add_argument('--mfcc', action='store_true',
                        help='Stage 2: Extract MFCC features (396x40)')
    parser.add_argument('--loso', action='store_true',
                        help='Stage 3: Create LOSO + train/val splits')
    parser.add_argument('--train', action='store_true',
                        help='Stage 4: Train ECAPA-TDNN models')
    parser.add_argument('--embeddings', action='store_true',
                        help='Stage 5: Extract embeddings from trained models')
    parser.add_argument('--train-plda', action='store_true',
                        help='Stage 6: Train PLDA models')
    parser.add_argument('--test-plda', action='store_true',
                        help='Stage 7: Test PLDA models')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--speaker', type=str, default=None,
                        help='Process specific speaker only (e.g., "03")')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main pipeline orchestrator."""
    args = parse_args()
    config = get_config(args.config)
    logger = setup_logging(args.log_dir)

    logger.info("=" * 70)
    logger.info("EMODB EMOTION RECOGNITION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Project root: {config['BASE_DIR']}")
    logger.info("=" * 70)

    run_all = args.all
    stages = {
        'segment':    args.segment    or run_all,
        'mfcc':       args.mfcc       or run_all,
        'loso':       args.loso       or run_all,
        'train':      args.train      or run_all,
        'embeddings': args.embeddings or run_all,
        'train_plda': args.train_plda or run_all,
        'test_plda':  args.test_plda  or run_all,
    }

    if not any(stages.values()):
        logger.error("No stages selected. Use --all or specify individual stages.")
        logger.error("Run 'python main.py --help' for usage.")
        sys.exit(1)

    active = [k for k, v in stages.items() if v]
    logger.info(f"Stages to run: {active}\n")

    start_time = datetime.now()

    try:
        if stages['segment']:
            segment_stage(config, logger)

        if stages['mfcc']:
            mfcc_stage(config, logger)

        if stages['loso']:
            loso_stage(config, logger)

        if stages['train']:
            train_stage(config, logger, speaker=args.speaker)

        if stages['embeddings']:
            embeddings_stage(config, logger, speaker=args.speaker)

        if stages['train_plda']:
            train_plda_stage(config, logger, speaker=args.speaker)

        if stages['test_plda']:
            test_plda_stage(config, logger, speaker=args.speaker)

        duration = datetime.now() - start_time
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Total duration: {duration}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("PIPELINE FAILED!")
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
