#!/usr/bin/env python3
"""
Train PLDA models for emotion recognition using embeddings from ECAPA-TDNN.

This script trains PLDA (Probabilistic Linear Discriminant Analysis) models
using embeddings extracted from trained ECAPA-TDNN models.

Workflow:
1. Extract embeddings from ECAPA-TDNN models (if not already done)
2. Apply optional LDA for dimensionality reduction
3. Train PLDA model for emotion classification

Usage:
    # Train single speaker
    python train_plda_models.py --speaker 03
    
    # Train all speakers
    python train_plda_models.py --all
    
    # Custom configuration
    python train_plda_models.py --speaker 03 --config config/plda_hparams.yaml
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils.training.plda_trainer import train_speaker_plda, train_all_speakers_plda
import hyperpyyaml


def load_config(config_path, emodb_config_path):
    """
    Load configuration files.
    
    Args:
        config_path: Path to PLDA hyperparameters YAML
        emodb_config_path: Path to main EmoDb configuration YAML
    
    Returns:
        config: Combined configuration dictionary
    """
    # Load PLDA hyperparameters
    with open(config_path, 'r') as f:
        plda_config = hyperpyyaml.load_hyperpyyaml(f)
    
    # Load main EmoDb config
    with open(emodb_config_path, 'r') as f:
        emodb_config = hyperpyyaml.load_hyperpyyaml(f)
    
    # Combine configs
    config = {
        'BASE_DIR': Path(emodb_config['BASE_DIR']),
        'PATHS': emodb_config['PATHS'],
        'PLDA': plda_config
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Train PLDA models for emotion recognition using ECAPA embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train speaker 03:
    python train_plda_models.py --speaker 03
  
  Train all speakers:
    python train_plda_models.py --all
  
  With custom embeddings directory:
    python train_plda_models.py --speaker 03 --embeddings-dir data/processed/embeddings
        """
    )
    
    # Training mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--speaker',
        type=str,
        help='Train single speaker (e.g., 03, 08, 15)'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Train all speakers'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/plda_hparams.yaml',
        help='Path to PLDA hyperparameters YAML file'
    )
    parser.add_argument(
        '--emodb-config',
        type=str,
        default='config/emodb_config.yaml',
        help='Path to main EmoDb configuration YAML file'
    )
    
    # PLDA parameters (override config file)
    parser.add_argument(
        '--use-lda',
        type=bool,
        default=None,
        help='Whether to use LDA dimensionality reduction before PLDA'
    )
    parser.add_argument(
        '--lda-dim',
        type=int,
        default=None,
        help='LDA output dimension (default: n_classes - 1)'
    )
    parser.add_argument(
        '--plda-dim',
        type=int,
        default=None,
        help='PLDA dimension (default: same as input)'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default=None,
        help='Directory containing pre-extracted embeddings'
    )
    parser.add_argument(
        '--embeddings-split',
        type=str,
        default='dev',
        help='Embeddings split name under each speaker (e.g., dev, other)'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    config = load_config(args.config, args.emodb_config)
    
    # Resolve paths
    base_dir = config['BASE_DIR']
    loso_dir = base_dir / config['PATHS']['LOSO']
    output_dir = base_dir / config['PLDA'].get('output_dir', 'output/models/plda')
    embeddings_dir = args.embeddings_dir or str(base_dir / config['PLDA'].get('embeddings_dir', 'data/embeddings'))
    
    # Get PLDA parameters (command line overrides config)
    use_lda = args.use_lda if args.use_lda is not None else config['PLDA'].get('use_lda', True)
    lda_dim = args.lda_dim or config['PLDA'].get('lda_dim')
    plda_dim = args.plda_dim or config['PLDA'].get('plda_dim')
    plda_iters = config['PLDA'].get('nb_iter', 10)
    
    print(f"Configuration:")
    print(f"  LOSO Directory: {loso_dir}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Embeddings Directory: {embeddings_dir}")
    print(f"  Embeddings Split: {args.embeddings_split}")
    print(f"  Use LDA: {use_lda}")
    print(f"  LDA Dimension: {lda_dim}")
    print(f"  PLDA Dimension: {plda_dim}")
    print(f"  PLDA Iterations: {plda_iters}")
    if use_lda or lda_dim:
        print("  Note: use_lda/lda_dim are ignored for SpeechBrain PLDA training")
    print()
    
    # Check if embeddings exist
    if not os.path.exists(embeddings_dir):
        print(f"✗ Embeddings directory not found: {embeddings_dir}")
        print(f"\nYou need to extract embeddings first:")
        print(f"  python utils/features_extraction/extract_embeddings.py --speaker {args.speaker or 'XX'}")
        print(f"\nOr for all speakers:")
        print(f"  python utils/features_extraction/extract_embeddings.py --all")
        sys.exit(1)
    
    # Train models
    if args.speaker:
        # Train single speaker
        print(f"Training PLDA model for speaker {args.speaker}...")
        
        # Check if embeddings exist for this speaker
        speaker_emb_dir = os.path.join(embeddings_dir, f"speaker_{args.speaker}")
        if not os.path.exists(speaker_emb_dir):
            print(f"✗ Embeddings not found for speaker {args.speaker}: {speaker_emb_dir}")
            print(f"\nExtract embeddings first:")
            print(f"  python utils/features_extraction/extract_embeddings.py --speaker {args.speaker}")
            sys.exit(1)
        
        try:
            results = train_speaker_plda(
                speaker_id=args.speaker,
                loso_dir=str(loso_dir),
                output_dir=str(output_dir),
                embeddings_dir=embeddings_dir,
                embeddings_split=args.embeddings_split,
                plda_dim=plda_dim,
                plda_iters=plda_iters
            )
            
            print(f"\n{'='*70}")
            print(f"Training Complete!")
            print(f"{'='*70}")
            print(f"Speaker: {args.speaker}")
            if results.get('val_accuracy') is not None:
                print(f"Validation Accuracy: {results['val_accuracy']:.4f} ({results['val_accuracy']*100:.2f}%)")
                print(f"Validation Error: {results['val_error']:.4f}")
            print(f"Model saved to: {results['model_path']}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Train all speakers
        print("Training PLDA models for all speakers...")
        
        try:
            all_results = train_all_speakers_plda(
                loso_dir=str(loso_dir),
                output_dir=str(output_dir),
                embeddings_dir=embeddings_dir,
                embeddings_split=args.embeddings_split,
                plda_dim=plda_dim,
                plda_iters=plda_iters
            )
            
            print(f"\n{'='*70}")
            print(f"All Training Complete!")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
