# ECAPA-TDNN Training for Emotion Recognition

This module provides utilities for training ECAPA-TDNN models for emotion recognition using SpeechBrain and LOSO (Leave-One-Speaker-Out) cross-validation.

## Files Created

### 1. **utils/training/ecapa_trainer.py** - Core Training Module

- `EmotionBrain`: Custom SpeechBrain Brain class for emotion recognition
- `prepare_datasets()`: Prepare train/validation datasets
- `train_speaker_model()`: Train model for single speaker
- `train_all_speakers()`: Train models for all speakers

### 2. **config/ecapa_hparams.yaml** - Model Configuration

- ECAPA-TDNN architecture parameters
- Training hyperparameters
- Optimizer and loss function settings

### 3. **train_ecapa_models.py** - Main Training Script

- Command-line interface for training
- Supports single speaker or all speakers training
- Saves results and models

## Usage

### Train All Speakers (LOSO)

```bash
python train_ecapa_models.py
```

### Train Single Speaker

```bash
python train_ecapa_models.py --speaker 03
```

### With Custom Parameters

```bash
python train_ecapa_models.py \
    --loso_dir data/processed/loso \
    --output_dir output/models \
    --hparams config/ecapa_hparams.yaml \
    --device cuda
```

### Programmatic Usage

```python
from utils.training import train_speaker_model, train_all_speakers
from hyperpyyaml import load_hyperpyyaml

# Load hyperparameters
with open('config/ecapa_hparams.yaml') as f:
    hparams = load_hyperpyyaml(f)

# Train single speaker
best_error = train_speaker_model(
    speaker_id='03',
    loso_dir='data/processed/loso',
    output_dir='output/models',
    hparams=hparams,
    run_opts={'device': 'cpu'}
)

# Or train all speakers
results = train_all_speakers(
    loso_dir='data/processed/loso',
    output_dir='output/models',
    hparams_file=hparams,
    run_opts={'device': 'cpu'}
)
```

## Architecture

### EmotionBrain Class

Inherits from `sb.core.Brain` and implements:

- `compute_forward()`: Load MFCC features and compute embeddings
- `compute_objectives()`: Calculate loss and update metrics
- `on_stage_start()`: Initialize metrics
- `on_stage_end()`: Update validation error

### Data Flow

1. Load MFCC features from .npy files
2. Normalize and pad sequences
3. Extract embeddings using ECAPA-TDNN
4. Classify emotions
5. Compute loss and update model

### Model Components

- **Mean-Variance Normalization**: Global normalization of features
- **ECAPA-TDNN**: Embedding extraction (192-dim embeddings)
- **Classifier**: Emotion classification (7 classes)
- **Loss**: Additive Angular Margin loss

## Output Structure

```bash
output/models/
├── speaker_03/
│   └── embedding_model.ckpt
├── speaker_08/
│   └── embedding_model.ckpt
├── speaker_09/
│   └── embedding_model.ckpt
...
└── training_results.json
```

## Training Results

The `training_results.json` file contains:

```json
{
  "03": {
    "status": "success",
    "best_error": 0.1234
  },
  "08": {
    "status": "success",
    "best_error": 0.1456
  }
}
```

## Hyperparameters

Key parameters in `ecapa_hparams.yaml`:

- **Batch size**: 8
- **Learning rate**: 0.001
- **Epochs**: 50
- **Embedding dimension**: 192
- **Number of classes**: 7 (emotions)
- **ECAPA-TDNN channels**: [1024, 1024, 1024, 1024, 3072]

## Requirements

```bash
torch
speechbrain
pandas
numpy
tqdm
hyperpyyaml
```

## Notes

- Uses LOSO cross-validation for speaker-independent evaluation
- Each speaker model is trained on data from all other speakers
- Validation is done on 20% of training data (stratified by emotion)
- Best model is saved based on validation error
- Features are expected in shape (40, time_frames) from MFCC extraction
