# EmoDb Emotion Recognition Project

A complete pipeline for emotion recognition using the Berlin Emotional Speech Database (EmoDb) and ECAPA-TDNN models.

## Overview

This project implements a comprehensive emotion recognition system that:

- Extracts metadata from EmoDb audio files
- Segments audio based on duration (2/3/4 parts)
- Extracts MFCC features (40 coefficients)
- Uses Leave-One-Speaker-Out (LOSO) cross-validation
- Trains ECAPA-TDNN models with SpeechBrain
- Achieves speaker-independent emotion recognition

## Features

✨ **Complete Pipeline**: From raw audio to trained models  
🔧 **Config-Based**: All parameters centralized in YAML  
📦 **Pip Installable**: `pip install -e .` for easy setup  
🎯 **LOSO Validation**: Speaker-independent evaluation  
🤖 **Multiple Models**: ECAPA-TDNN and LDA support  
📊 **Comprehensive Logging**: Track all pipeline stages  
🚀 **Multiple Entry Points**: CLI, Python API, and shell scripts  

## Quick Start

### 1. Installation

```bash
# Clone repository
cd /path/to/emodb_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install package
pip install -e .

# Verify installation
python verify_installation.py
```

### 2. Run Complete Pipeline

```bash
# Easiest: Use run.sh
./run.sh all

# Or use main.py
python main.py --all

# Or step by step
./run.sh preprocess  # metadata, segment, MFCC
./run.sh prepare     # LOSO, train/val splits
./run.sh train       # train models
```

### 3. Train Models

**ECAPA-TDNN (Deep Learning):**

```bash
# Train all speakers
./run.sh train

# Train specific speaker
./run.sh train-speaker 03

# Using Python directly
python train_ecapa_models.py --speaker 03
```

**LDA (Statistical Model):**

```bash
# Train all speakers
./run.sh train-lda

# Train specific speaker
./run.sh train-lda-speaker 03

# Using Python directly
python train_lda_models.py --speaker 03
```

## Project Structure

```bash
emodb_project/
├── main.py                    # Main pipeline orchestrator
├── train_ecapa_models.py      # ECAPA-TDNN training script
├── train_lda_models.py       # LDA training script
├── run.sh                     # Convenience shell script
├── verify_installation.py     # Installation verification
├── pyproject.toml            # Package configuration
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
│
├── config/
│   ├── emodb_config.yaml     # Main configuration
│   ├── ecapa_hparams.yaml    # ECAPA-TDNN hyperparameters
│   └── lda_hparams.yaml     # LDA hyperparameters
│
├── utils/                     # Core utilities package
│   ├── __init__.py
│   ├── extract_config.py     # Config management
│   ├── audio_processing/     # Audio processing modules
│   │   ├── extract_metadata.py
│   │   └── create_csv.py
│   ├── features_extraction/  # Feature extraction modules
│   │   ├── extract_mfcc.py
│   │   ├── create_loso_splits.py
│   │   └── create_train_val_splits.py
│   └── training/             # Training modules
│       ├── ecapa_trainer.py
│       └── lda_trainer.py
│
├── data/
│   ├── raw/emodb/            # Raw audio files (.wav)
│   ├── csv/                  # Metadata CSVs
│   │   ├── metadata.csv
│   │   └── segmented_metadata.csv
│   └── processed/
│       ├── segment/          # Segmented audio
│       ├── features/         # MFCC .npy files
│       └── loso/            # LOSO splits
│           └── speaker_XX/
│               ├── dev/      # 80% training data
│               ├── train/    # 20% validation data
│               ├── other/    # Other speakers' data
│               └── test/     # Test speaker's data
│
├── output/
│   ├── logs/                 # Pipeline logs
│   ├── models/              # Trained models
│   │   ├── speaker_XX/      # ECAPA-TDNN checkpoints per speaker
│   │   ├── lda/            # LDA models
│   │   │   └── speaker_XX/  # LDA per speaker
│   │   ├── training_results.json  # All speakers ECAPA results
│   │   ├── training_results_speaker_XX.json  # Single speaker ECAPA results
│   │   └── lda_training_results.json  # LDA results
│   └── results/             # Evaluation results
│
├── scripts/
│   └── archive/             # Old scripts (archived)
│
└── notebooks/               # Jupyter notebooks
    └── 01_data_processing.ipynb
```

## Pipeline Stages

### Stage 1: Metadata Extraction

Extracts speaker ID, emotion, duration, and samples from audio filenames.

```bash
./run.sh metadata
```

### Stage 2: Audio Segmentation

Segments audio based on duration:

- Duration ≥ 6s → 4 parts
- Duration ≥ 4s → 3 parts  
- Duration ≥ 2s → 2 parts
- Duration < 2s → 1 part (no segmentation)

```bash
./run.sh segment
```

### Stage 3: MFCC Feature Extraction

Extracts 40 MFCC coefficients with configurable parameters:

- n_fft: 400 (25ms window @ 16kHz)
- hop_length: 80 (5ms hop)
- n_mels: 40

```bash
./run.sh mfcc
```

Alternative (if MFCCs already extracted in MATLAB):

```bash
# Build emodb_mfcc_features.csv from existing .npy files
./run.sh create-mfcc-csv
```

This assumes your MFCC .npy files are in the configured path
(`MFCC.OUTPUT_DIR`, default `data/processed/features/mfcc`).

### Stage 4: LOSO Split Creation

Creates Leave-One-Speaker-Out splits for cross-validation.

```bash
./run.sh loso
```

### Stage 5: Train/Validation Splits

Creates stratified 80/20 splits within each speaker's training data.

```bash
./run.sh splits
```

### Stage 6: Model Training

Train emotion recognition models using different approaches:

**ECAPA-TDNN (Deep Learning):**

```bash
# Train all speakers
./run.sh train

# Train specific speaker
./run.sh train-speaker 03
```

**LDA (Statistical Model):**

```bash
# Train all speakers
./run.sh train-lda

# Train specific speaker
./run.sh train-lda-speaker 03
```

## Usage Examples

### Complete Pipeline

```bash
# Run everything at once
python main.py --all

# With custom config
python main.py --all --config config/custom_config.yaml
```

### Preprocessing Only

```bash
# All preprocessing stages
python main.py --metadata --segment --mfcc

# Or individual stages
python main.py --metadata
python main.py --segment
python main.py --mfcc
```

### Embedding Averaging (Testing Prep)

Average train embeddings per emotion (7 centroids) and save to
`data/testing/speaker_{id}/train` with a summary CSV at
`data/testing/speaker_{id}/emotion_centroids.csv`.

```bash
./run.sh avg-embeddings
./run.sh avg-embeddings-speaker 03
```

Average test embeddings by base id (e.g., `abc_1.npy`, `abc_2.npy` → `abc.npy`)
and save to `data/testing/speaker_{id}/test`. The summary CSV is stored at
`data/testing/speaker_{id}/test_averaged_embeddings.csv` with columns:
`id`, `embedding_path`, `emotion_path`, `ground_truth` (one-hot per emotion).

```bash
./run.sh avg-test-embeddings
./run.sh avg-test-embeddings-speaker 03
```

### Training Workflows

**ECAPA-TDNN:**

```bash
# Train all speakers
python train_ecapa_models.py --all

# Train specific speaker
python train_ecapa_models.py --speaker 03

# With custom paths
python train_ecapa_models.py --speaker 03 --loso_dir path/to/loso --output_dir path/to/output
```

**LDA:**

```bash
# Train all speakers
python train_lda_models.py --all

# Train specific speaker
python train_lda_models.py --speaker 03

# With custom parameters
python train_lda_models.py --speaker 03 --n-components 5 --solver eigen --shrinkage auto

# Different feature column
python train_lda_models.py --speaker 03 --feature-column mfcc
```

### Using Python API

```python
from utils import get_config
from utils.audio_processing import extract_metadata_from_folder
from utils.features_extraction import extract_mfcc_from_dataset
from utils.training import train_speaker_model

# Load configuration
config = get_config()

# Extract metadata
metadata = extract_metadata_from_folder(
    folder_path="data/raw/emodb",
    config=config
)

# Extract MFCC features
successful, failed = extract_mfcc_from_dataset(
    csv_path="data/csv/segmented_metadata.csv",
    output_dir="data/processed/features",
    config=config
)

# Train model
best_error = train_speaker_model(
    speaker_id="03",
    loso_dir="data/processed/loso",
    output_dir="output/models"
)
```

## Configuration

All parameters are centralized in [config/emodb_config.yaml](config/emodb_config.yaml):

```yaml
PATHS:
  project_root: /path/to/emodb_project
  raw_audio: data/raw/emodb
  csv: data/csv
  features: data/processed/features
  loso: data/processed/loso
  output: output

AUDIO:
  sampling_rate: 16000

EMOTION_MAPPING:
  W: 2  # Anger
  L: 4  # Boredom
  E: 5  # Disgust
  A: 3  # Fear/Anxiety
  F: 0  # Happiness
  T: 6  # Sadness
  N: 1  # Neutral

MFCC:
  n_mfcc: 40
  n_fft: 400
  hop_length: 80
  # ... other parameters

LOSO:
  train_ratio: 0.8
  random_state: 42
```

## Data Format

### EmoDb Filename Convention

Format: `{speaker_id}{emotion}{sentence_id}{version}.wav`

Example: `03a01Fa.wav`

- Speaker: 03
- Emotion: a (Anger → code 2)
- Sentence: 01
- Version: F.a

### Emotion Labels

- **W (Wut)**: Anger → 2
- **L (Langeweile)**: Boredom → 4
- **E (Ekel)**: Disgust → 5
- **A (Angst)**: Fear/Anxiety → 3
- **F (Freude)**: Happiness → 0
- **T (Trauer)**: Sadness → 6
- **N (Neutral)**: Neutral → 1

## Model Architecture

### ECAPA-TDNN (Deep Learning)

- **Input**: 40-dimensional MFCC features
- **Architecture**: Time Delay Neural Network with channel attention
- **Output**: 7-class emotion classification
- **Framework**: SpeechBrain 1.0.3
- **Optimizer**: Adam
- **Loss**: NLL (Negative Log-Likelihood)

**Hyperparameters** (in [config/ecapa_hparams.yaml](config/ecapa_hparams.yaml)):

- Input size: 40
- Embedding size: 192
- Number of epochs: 2 (adjust as needed)
- Batch size: 8
- Learning rate: 0.001

### LDA (Statistical Model)

- **Input**: Statistical features from MFCC (mean, std, min, max)
- **Algorithm**: Probabilistic Linear Discriminant Analysis
- **Output**: 7-class emotion classification
- **Framework**: scikit-learn
- **Preprocessing**: Feature standardization (automatic)
- **Advantages**: Fast training, no GPU needed, interpretable

**Hyperparameters** (in [config/lda_hparams.yaml](config/lda_hparams.yaml)):

- n_components: None (defaults to min(n_classes-1, n_features))
- solver: 'svd' (also supports 'lsqr', 'eigen')
- shrinkage: None (regularization for 'lsqr'/'eigen' solvers)
- feature_column: 'mfcc'

**Feature Extraction for LDA:**

For each MFCC file (time × features), LDA uses:
- Mean across time
- Standard deviation across time
- Minimum across time
- Maximum across time

This reduces variable-length sequences to fixed-size feature vectors.

## Results

Training results are saved to JSON files:

**ECAPA-TDNN Results:**

```bash
# Single speaker
output/models/training_results_speaker_03.json

# All speakers
output/models/training_results.json
```

Example ECAPA-TDNN format:

```json
{
  "speaker_03": {
    "status": "success",
    "best_error": 0.1234,
    "model_path": "output/models/speaker_03"
  }
}
```

**LDA Results:**

```bash
# Single speaker
output/models/lda/speaker_03/lda_results.json

# All speakers
output/models/lda/lda_training_results.json
```

Example LDA format:

```json
{
  "speaker": "03",
  "model_path": "output/models/lda/speaker_03/lda_model.pkl",
  "train_accuracy": 0.9876,
  "train_error": 0.0124,
  "val_accuracy": 0.8543,
  "val_error": 0.1457,
  "n_train_samples": 450,
  "n_val_samples": 112,
  "n_components": 6,
  "feature_dim": 160
}
```

## Logging

All pipeline runs generate detailed logs:

- **Location**: `output/logs/pipeline_YYYYMMDD_HHMMSS.log`
- **Format**: Timestamped with log levels
- **Contents**: Stage progress, errors, warnings

```bash
# View latest log
ls -t output/logs/*.log | head -1 | xargs tail -f
```

## Troubleshooting

### Import Errors

```bash
pip install -e .
python verify_installation.py
```

### Missing Data

```bash
# Run preprocessing first
./run.sh preprocess
```

### Training Failures

```bash
# Check logs
tail -f output/logs/*.log

# Try single speaker first
./run.sh train-speaker 03
```

### Path Issues

```bash
# Verify config paths
cat config/emodb_config.yaml | grep PATHS
```

## Development

### Running Tests

```bash
# Verify installation
python verify_installation.py

# Test single speaker
./run.sh train-speaker 03
```

### Adding New Features

1. Update configuration in `config/emodb_config.yaml`
2. Modify relevant modules in `utils/`
3. Test with single speaker
4. Update documentation

### Code Quality

- All functions have comprehensive docstrings
- Configuration-based (no hardcoded paths)
- Type hints where appropriate
- Error handling throughout

## Requirements

- Python 3.13+
- PyTorch 2.0+
- torchaudio 2.0+
- SpeechBrain 1.0+
- librosa, pandas, numpy, etc.

See [requirements.txt](requirements.txt) for complete list.

## License

[Add your license here]

## Citation

If you use this code, please cite:

```bash
[Add citation information]
```

## Acknowledgments

- Berlin Emotional Speech Database (EmoDb)
- SpeechBrain framework
- ECAPA-TDNN architecture

## Contact

[Add contact information]
