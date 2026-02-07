#!/usr/bin/env python3
"""
EmoDb Emotion Recognition Pipeline - Main Entry Point

This script orchestrates the complete emotion recognition pipeline:
1. Metadata extraction from audio files
2. Audio segmentation
3. MFCC feature extraction
4. LOSO (Leave-One-Speaker-Out) split creation
5. Train/validation split creation (80/20)
6. Model training with ECAPA-TDNN
7. Model evaluation

Usage:
    # Run complete pipeline
    python main.py --all
    
    # Run specific stages
    python main.py --metadata --segment --mfcc
    python main.py --loso --splits
    python main.py --train --speaker 03
    
    # Run with custom config
    python main.py --all --config path/to/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import utils
from utils import get_config
from utils.audio_processing import (
    extract_metadata_from_folder,
    extract_segment_from_folder,
    create_csv
)
from utils.features_extraction import (
    extract_mfcc_from_dataset,
    create_loso_splits,
    create_train_val_splits
)
from utils.training import train_all_speakers, train_speaker_model


def setup_logging(log_dir=None):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files. If None, uses config default.
    
    Returns:
        Logger instance
    """
    if log_dir is None:
        config = get_config()
        log_dir = Path(config['BASE_DIR']) / config['PATHS']['LOGS']
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    # Configure logging
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


def extract_metadata_stage(config, logger):
    """
    Stage 1: Extract metadata from audio files.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("STAGE 1: METADATA EXTRACTION")
    logger.info("="*70)
    
    raw_audio_dir = Path(config['BASE_DIR']) / config['PATHS']['RAW_DATA']
    output_csv = Path(config['BASE_DIR']) / config['PATHS']['CSV'] / 'metadata.csv'
    
    logger.info(f"Input directory: {raw_audio_dir}")
    logger.info(f"Output CSV: {output_csv}")
    
    # Extract metadata
    metadata = extract_metadata_from_folder(
        folder_path=str(raw_audio_dir),
        config=config
    )
    
    logger.info(f"Extracted metadata for {len(metadata)} files")
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    create_csv(metadata, str(output_csv))
    
    logger.info(f"Metadata saved to: {output_csv}")
    logger.info("Stage 1 complete!\n")
    
    return metadata


def segment_audio_stage(config, logger):
    """
    Stage 2: Segment audio files.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("STAGE 2: AUDIO SEGMENTATION")
    logger.info("="*70)
    
    raw_audio_dir = Path(config['BASE_DIR']) / config['PATHS']['RAW_DATA']
    segment_dir = Path(config['BASE_DIR']) / config['PATHS']['SEGMENT']
    output_csv = Path(config['BASE_DIR']) / config['PATHS']['CSV'] / 'segmented_metadata.csv'
    
    logger.info(f"Input directory: {raw_audio_dir}")
    logger.info(f"Segment directory: {segment_dir}")
    logger.info(f"Output CSV: {output_csv}")
    
    # Segment audio
    segmented_metadata = extract_segment_from_folder(
        folder_path=str(raw_audio_dir),
        output_folder=str(segment_dir),
        config=config
    )
    
    logger.info(f"Created {len(segmented_metadata)} segments")
    
    # Save to CSV
    create_csv(segmented_metadata, str(output_csv))
    
    logger.info(f"Segmented metadata saved to: {output_csv}")
    logger.info("Stage 2 complete!\n")
    
    return segmented_metadata


def extract_mfcc_stage(config, logger):
    """
    Stage 3: Extract MFCC features.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("STAGE 3: MFCC FEATURE EXTRACTION")
    logger.info("="*70)
    
    input_csv = Path(config['BASE_DIR']) / config['PATHS']['CSV'] / 'segmented_metadata.csv'
    features_dir = Path(config['BASE_DIR']) / config['PATHS']['FEATURES']
    
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Features directory: {features_dir}")
    
    # MFCC parameters
    mfcc_params = config.get('MFCC', {})
    logger.info(f"MFCC parameters: {mfcc_params}")
    
    # Extract MFCC features
    successful, failed = extract_mfcc_from_dataset(
        csv_path=str(input_csv),
        output_dir=str(features_dir),
        config=config
    )
    
    logger.info(f"Successfully extracted features: {successful}/{successful + failed}")
    if failed > 0:
        logger.warning(f"Failed extractions: {failed}")
    
    logger.info("Stage 3 complete!\n")
    
    return successful, failed


def create_loso_stage(config, logger):
    """
    Stage 4: Create LOSO (Leave-One-Speaker-Out) splits.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("STAGE 4: LOSO SPLIT CREATION")
    logger.info("="*70)
    
    input_csv = Path(config['BASE_DIR']) / config['PATHS']['CSV'] / 'segmented_metadata.csv'
    loso_dir = Path(config['BASE_DIR']) / config['PATHS']['LOSO']
    
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"LOSO directory: {loso_dir}")
    
    # Create LOSO splits
    speakers = create_loso_splits(
        input_csv=str(input_csv),
        output_dir=str(loso_dir),
        config=config
    )
    
    logger.info(f"Created LOSO splits for {len(speakers)} speakers: {sorted(speakers)}")
    logger.info("Stage 4 complete!\n")
    
    return speakers


def create_train_val_splits_stage(config, logger):
    """
    Stage 5: Create train/validation splits (80/20).
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("STAGE 5: TRAIN/VALIDATION SPLIT CREATION")
    logger.info("="*70)
    
    loso_dir = Path(config['BASE_DIR']) / config['PATHS']['LOSO']
    
    logger.info(f"LOSO directory: {loso_dir}")
    
    # Get train ratio from config
    train_ratio = config.get('LOSO', {}).get('train_ratio', 0.8)
    logger.info(f"Train ratio: {train_ratio}")
    
    # Create train/val splits for all speakers
    speakers = sorted([d.name for d in loso_dir.iterdir() if d.is_dir() and d.name.startswith('speaker_')])
    
    logger.info(f"Creating train/val splits for {len(speakers)} speakers")
    
    for speaker in speakers:
        logger.info(f"Processing {speaker}...")
        speaker_dir = loso_dir / speaker
        
        # Create train/val splits
        create_train_val_splits(
            loso_dir=str(speaker_dir),
            config=config
        )
    
    logger.info("Stage 5 complete!\n")
    
    return speakers


def train_models_stage(config, logger, speaker=None):
    """
    Stage 6: Train ECAPA-TDNN models.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        speaker: Specific speaker ID to train (e.g., "03"). If None, train all.
    """
    logger.info("="*70)
    logger.info("STAGE 6: MODEL TRAINING")
    logger.info("="*70)
    
    loso_dir = Path(config['BASE_DIR']) / config['PATHS']['LOSO']
    output_dir = Path(config['BASE_DIR']) / config['PATHS']['MODELS']
    hparams_file = Path(config['BASE_DIR']) / 'config' / 'ecapa_hparams.yaml'
    
    logger.info(f"LOSO directory: {loso_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Hyperparameters: {hparams_file}")
    
    # Load hyperparameters
    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    run_opts = {"device": "cpu"}  # Change to "cuda" if GPU available
    
    if speaker:
        # Train single speaker
        logger.info(f"Training single speaker: {speaker}")
        
        best_error = train_speaker_model(
            speaker_id=speaker,
            loso_dir=str(loso_dir),
            output_dir=str(output_dir),
            hparams=hparams,
            run_opts=run_opts
        )
        
        logger.info(f"Training complete for speaker {speaker}")
        logger.info(f"Best validation error: {best_error:.4f}")
        
        # Save single speaker results to JSON
        import json
        results = {
            f"speaker_{speaker}": {
                "status": "success",
                "best_error": float(best_error),
                "model_path": str(output_dir / f"speaker_{speaker}" / "CKPT+*")
            }
        }
        results_file = output_dir / f'training_results_speaker_{speaker}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Model saved to: {output_dir / f'speaker_{speaker}'}")
        
    else:
        # Train all speakers
        logger.info("Training all speakers...")
        
        results = train_all_speakers(
            loso_dir=str(loso_dir),
            output_dir=str(output_dir),
            hparams_file=hparams,
            run_opts=run_opts
        )
        
        # Save results
        import json
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        successful = [s for s, r in results.items() if r['status'] == 'success']
        failed = [s for s, r in results.items() if r['status'] == 'failed']
        
        logger.info(f"Successful: {len(successful)}/{len(results)}")
        logger.info(f"Failed: {len(failed)}/{len(results)}")
        
        if successful:
            avg_error = sum(results[s]['best_error'] for s in successful) / len(successful)
            logger.info(f"Average validation error: {avg_error:.4f}")
        
        logger.info(f"Results saved to: {results_file}")
    
    logger.info("Stage 6 complete!\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EmoDb Emotion Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --all
  
  # Run preprocessing only
  python main.py --metadata --segment --mfcc
  
  # Run data preparation
  python main.py --loso --splits
  
  # Train all models
  python main.py --train
  
  # Train specific speaker
  python main.py --train --speaker 03
  
  # Use custom config
  python main.py --all --config config/custom_config.yaml
        """
    )
    
    # Stage selection
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (all stages)')
    parser.add_argument('--metadata', action='store_true',
                       help='Stage 1: Extract metadata')
    parser.add_argument('--segment', action='store_true',
                       help='Stage 2: Segment audio files')
    parser.add_argument('--mfcc', action='store_true',
                       help='Stage 3: Extract MFCC features')
    parser.add_argument('--loso', action='store_true',
                       help='Stage 4: Create LOSO splits')
    parser.add_argument('--splits', action='store_true',
                       help='Stage 5: Create train/val splits')
    parser.add_argument('--train', action='store_true',
                       help='Stage 6: Train models')
    
    # Options
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--speaker', type=str, default=None,
                       help='Train specific speaker only (e.g., "03")')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files')
    
    return parser.parse_args()


def main():
    """Main pipeline orchestrator."""
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    logger.info("="*70)
    logger.info("EMODB EMOTION RECOGNITION PIPELINE")
    logger.info("="*70)
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Project root: {config['BASE_DIR']}")
    logger.info("="*70)
    logger.info("")
    
    # Determine which stages to run
    run_all = args.all
    run_metadata = args.metadata or run_all
    run_segment = args.segment or run_all
    run_mfcc = args.mfcc or run_all
    run_loso = args.loso or run_all
    run_splits = args.splits or run_all
    run_train = args.train or run_all
    
    # Check if any stage is selected
    if not any([run_metadata, run_segment, run_mfcc, run_loso, run_splits, run_train]):
        logger.error("No stages selected! Use --all or specify individual stages.")
        logger.error("Run 'python main.py --help' for usage information.")
        sys.exit(1)
    
    # Track pipeline progress
    start_time = datetime.now()
    
    try:
        # Stage 1: Metadata extraction
        if run_metadata:
            extract_metadata_stage(config, logger)
        
        # Stage 2: Audio segmentation
        if run_segment:
            segment_audio_stage(config, logger)
        
        # Stage 3: MFCC feature extraction
        if run_mfcc:
            extract_mfcc_stage(config, logger)
        
        # Stage 4: LOSO split creation
        if run_loso:
            create_loso_stage(config, logger)
        
        # Stage 5: Train/val split creation
        if run_splits:
            create_train_val_splits_stage(config, logger)
        
        # Stage 6: Model training
        if run_train:
            train_models_stage(config, logger, speaker=args.speaker)
        
        # Pipeline complete
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total duration: {duration}")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error("="*70)
        logger.error("PIPELINE FAILED!")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
