#!/usr/bin/env python3
"""
Train ECAPA-TDNN models for emotion recognition using LOSO cross-validation.

Usage:
    python train_ecapa_models.py [--speaker ID] [--config PATH]
    
Examples:
    # Train all speakers
    python train_ecapa_models.py
    
    # Train specific speaker
    python train_ecapa_models.py --speaker 03
    
    # Use custom config
    python train_ecapa_models.py --config config/custom_config.yaml
"""
import os
import sys
import argparse
import json
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

from utils import get_config
from utils.training import (
    train_all_speakers,
    train_speaker_model,
    train_all_speakers_other,
    train_speaker_model_other,
    train_all_speakers_firstpart,
    train_speaker_model_firstpart,
    train_all_speakers_other_firstpart,
    train_speaker_model_other_firstpart
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ECAPA-TDNN models for emotion recognition"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/emodb_config.yaml)'
    )
    
    parser.add_argument(
        '--loso_dir',
        type=str,
        default=None,
        help='Path to LOSO splits directory (overrides config)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for trained models (overrides config)'
    )
    
    parser.add_argument(
        '--hparams',
        type=str,
        default=None,
        help='Path to hyperparameters YAML file (default: config/ecapa_hparams.yaml)'
    )
    
    parser.add_argument(
        '--speaker',
        type=str,
        default=None,
        help='Train only specific speaker (e.g., "03"). If not provided, train all.'
    )

    parser.add_argument(
        '--train-other',
        action='store_true',
        help='Train using other.csv (all remaining speakers)'
    )

    parser.add_argument(
        '--first-part',
        action='store_true',
        help='Train using only the first segment for each file'
    )

    parser.add_argument(
        '--no-valid',
        action='store_true',
        help='Skip validation when training on other.csv'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu/cuda)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Get paths from config or arguments
    base_dir = Path(config['BASE_DIR'])
    loso_dir = args.loso_dir or str(base_dir / config['PATHS']['LOSO'])
    if args.train_other and args.output_dir is None:
        output_dir = str(base_dir / 'output' / 'models_other')
    else:
        output_dir = args.output_dir or str(base_dir / config['PATHS']['MODELS'])
    hparams_file = args.hparams or str(base_dir / 'config' / 'ecapa_hparams.yaml')
    
    # Load hyperparameters
    print(f"Loading hyperparameters from: {hparams_file}")
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    # Set run options
    run_opts = {"device": args.device}
    
    separator = '=' * 70
    print(f"\n{separator}")
    print("ECAPA-TDNN Emotion Recognition Training")
    print(separator)
    print(f"Configuration: {args.config or 'default'}")
    print(f"LOSO directory: {loso_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"{separator}\n")
    
    # Train models
    if args.speaker:
        # Train single speaker
        print(f"Training single speaker: {args.speaker}")

        if args.train_other:
            if args.first_part:
                best_error = train_speaker_model_other_firstpart(
                    speaker_id=args.speaker,
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams=hparams,
                    run_opts=run_opts,
                    use_validation=not args.no_valid,
                )
            else:
                best_error = train_speaker_model_other(
                    speaker_id=args.speaker,
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams=hparams,
                    run_opts=run_opts,
                    use_validation=not args.no_valid,
                )
        else:
            if args.first_part:
                best_error = train_speaker_model_firstpart(
                    speaker_id=args.speaker,
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams=hparams,
                    run_opts=run_opts
                )
            else:
                best_error = train_speaker_model(
                    speaker_id=args.speaker,
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams=hparams,
                    run_opts=run_opts
                )
        
        print(f"\n{separator}")
        print(f"Training complete for speaker {args.speaker}")
        if best_error is not None:
            print(f"Best validation error: {best_error:.4f}")
        else:
            print("Validation skipped")
        print(f"{separator}\n")
        
        # Save single speaker results to JSON
        results = {
            f"speaker_{args.speaker}": {
                "status": "success",
                "best_error": float(best_error),
                "model_path": os.path.join(output_dir, f"speaker_{args.speaker}")
            }
        }
        results_file = os.path.join(output_dir, f'training_results_speaker_{args.speaker}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        print(f"Model saved to: {os.path.join(output_dir, f'speaker_{args.speaker}')}")
        
    else:
        # Train all speakers
        print("Training all speakers...")
        
        if args.train_other:
            if args.first_part:
                results = train_all_speakers_other_firstpart(
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams_file=hparams,
                    run_opts=run_opts,
                    use_validation=not args.no_valid,
                )
            else:
                results = train_all_speakers_other(
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams_file=hparams,
                    run_opts=run_opts,
                    use_validation=not args.no_valid,
                )
        else:
            if args.first_part:
                results = train_all_speakers_firstpart(
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams_file=hparams,
                    run_opts=run_opts
                )
            else:
                results = train_all_speakers(
                    loso_dir=loso_dir,
                    output_dir=output_dir,
                    hparams_file=hparams,
                    run_opts=run_opts
                )
        
        # Save results
        results_file = os.path.join(output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{separator}")
        print("Training Summary")
        print(separator)
        
        successful = [s for s, r in results.items() if r['status'] == 'success']
        failed = [s for s, r in results.items() if r['status'] == 'failed']
        
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Failed: {len(failed)}/{len(results)}")
        
        if successful:
            best_errors = [results[s]['best_error'] for s in successful if results[s]['best_error'] is not None]
            if best_errors:
                avg_error = sum(best_errors) / len(best_errors)
                print(f"\nAverage validation error: {avg_error:.4f}")
            else:
                print("\nValidation skipped for all speakers")
            
            print("\nPer-speaker results:")
            for speaker in sorted(successful):
                error = results[speaker]['best_error']
                print(f"  Speaker {speaker}: {error:.4f}")
        
        if failed:
            print(f"\nFailed speakers:")
            for speaker in failed:
                error_msg = results[speaker]['error']
                print(f"  Speaker {speaker}: {error_msg}")
        
        print(f"\nResults saved to: {results_file}")
        print(f"Models saved to: {output_dir}")
        print(f"{separator}\n")


if __name__ == "__main__":
    main()
