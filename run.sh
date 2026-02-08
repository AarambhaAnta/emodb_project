#!/bin/bash
# EmoDb Emotion Recognition Pipeline - Quick Run Script
# 
# This script provides convenient commands to run different parts of the pipeline.
# Usage: ./run.sh [command]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Function to print colored output
print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Show usage
usage() {
    echo "EmoDb Emotion Recognition Pipeline - Quick Run Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all              - Run complete pipeline (preprocessing + training)"
    echo "  preprocess       - Run preprocessing only (metadata, segment, MFCC)"
    echo "  prepare          - Prepare data (LOSO + train/val splits)"
    echo "  train            - Train all ECAPA-TDNN models"
    echo "  train-speaker N  - Train ECAPA-TDNN for specific speaker (e.g., train-speaker 03)"
    echo "  train-lda        - Train all LDA models"
    echo "  train-lda-speaker N - Train LDA for specific speaker (e.g., train-lda-speaker 03)"
    echo "  train-plda       - Train all PLDA models"
    echo "  train-plda-speaker N - Train PLDA for specific speaker (e.g., train-plda-speaker 03)"
    echo "  train-plda-dev     - Train PLDA on dev embeddings for all speakers"
    echo "  train-plda-dev-speaker N - Train PLDA on dev embeddings for specific speaker"
    echo "  extract-embeddings - Extract embeddings for all speakers"
    echo "  extract-embeddings-speaker N - Extract embeddings for specific speaker"
    echo "  avg-embeddings     - Average train embeddings per emotion for all speakers"
    echo "  avg-embeddings-speaker N - Average train embeddings for a speaker"
    echo "  avg-test-embeddings     - Average test embeddings by base id for all speakers"
    echo "  avg-test-embeddings-speaker N - Average test embeddings by base id for a speaker"
    echo "  score-plda        - PLDA scoring for all speakers"
    echo "  score-plda-speaker N - PLDA scoring for a speaker"
    echo "  metadata         - Extract metadata only"
    echo "  segment          - Segment audio only"
    echo "  mfcc             - Extract MFCC features only"
    echo "  create-mfcc-csv  - Create MFCC features CSV from .npy files"
    echo "  loso             - Create LOSO splits only"
    echo "  splits           - Create train/val splits only"
    echo "  verify           - Verify installation"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh all                # Run complete pipeline"
    echo "  ./run.sh preprocess         # Preprocessing only"
    echo "  ./run.sh create-mfcc-csv    # Create MFCC CSV from .npy files"
    echo "  ./run.sh train-speaker 03   # Train ECAPA-TDNN for speaker 03"
    echo "  ./run.sh train-lda          # Train all LDA models"
    echo "  ./run.sh train-lda-speaker 03  # Train LDA for speaker 03"
    echo "  ./run.sh extract-embeddings-speaker 03  # Extract embeddings for speaker 03"
    echo "  ./run.sh train-plda-speaker 03  # Train PLDA for speaker 03"
    echo "  ./run.sh train-plda-dev        # Train PLDA on dev embeddings"
    echo "  ./run.sh train-plda-dev-speaker 03  # Train PLDA on dev embeddings for speaker 03"
    echo "  ./run.sh avg-embeddings    # Average train embeddings per emotion"
    echo "  ./run.sh avg-embeddings-speaker 03  # Average train embeddings for speaker 03"
    echo "  ./run.sh avg-test-embeddings    # Average test embeddings by base id"
    echo "  ./run.sh avg-test-embeddings-speaker 03  # Average test embeddings for speaker 03"
    echo "  ./run.sh score-plda    # PLDA scoring for all speakers"
    echo "  ./run.sh score-plda-speaker 03  # PLDA scoring for speaker 03"
}

# Parse command
COMMAND="${1:-help}"

case "$COMMAND" in
    all)
        print_header "Running Complete Pipeline"
        python main.py --all
        print_success "Complete pipeline finished!"
        ;;
    
    preprocess)
        print_header "Running Preprocessing Pipeline"
        python main.py --metadata --segment --mfcc
        print_success "Preprocessing complete!"
        ;;
    
    prepare)
        print_header "Preparing Data (LOSO + Splits)"
        python main.py --loso --splits
        print_success "Data preparation complete!"
        ;;
    
    train)
        print_header "Training All Models"
        python main.py --train
        print_success "Training complete!"
        ;;
    
    train-speaker)
        SPEAKER="${2:-03}"
        print_header "Training ECAPA-TDNN for Speaker $SPEAKER"
        python main.py --train --speaker "$SPEAKER"
        print_success "ECAPA-TDNN training complete for speaker $SPEAKER!"
        ;;
    
    train-lda)
        print_header "Training LDA Models for All Speakers"
        python train_lda_models.py --all
        print_success "LDA training complete!"
        ;;
    
    train-lda-speaker)
        SPEAKER="${2:-03}"
        print_header "Training LDA for Speaker $SPEAKER"
        python train_lda_models.py --speaker "$SPEAKER"
        print_success "LDA training complete for speaker $SPEAKER!"
        ;;
    
    extract-embeddings)
        print_header "Extracting Embeddings for All Speakers"
        python utils/features_extraction/extract_embeddings.py --all
        print_success "Embedding extraction complete!"
        ;;
    
    extract-embeddings-speaker)
        SPEAKER="${2:-03}"
        print_header "Extracting Embeddings for Speaker $SPEAKER"
        python utils/features_extraction/extract_embeddings.py --speaker "$SPEAKER"
        print_success "Embedding extraction complete for speaker $SPEAKER!"
        ;;
    
    train-plda)
        print_header "Training PLDA Models for All Speakers"
        python train_plda_models.py --all
        print_success "PLDA training complete!"
        ;;
    
    train-plda-speaker)
        SPEAKER="${2:-03}"
        print_header "Training PLDA for Speaker $SPEAKER"
        python train_plda_models.py --speaker "$SPEAKER"
        print_success "PLDA training complete for speaker $SPEAKER!"
        ;;

    train-plda-dev)
        print_header "Training PLDA Models (Dev Embeddings)"
        python train_plda_models.py --all --embeddings-dir data/embeddings
        print_success "PLDA training complete (dev embeddings)!"
        ;;

    train-plda-dev-speaker)
        SPEAKER="${2:-03}"
        print_header "Training PLDA for Speaker $SPEAKER (Dev Embeddings)"
        python train_plda_models.py --speaker "$SPEAKER" --embeddings-dir data/embeddings
        print_success "PLDA training complete for speaker $SPEAKER (dev embeddings)!"
        ;;

    avg-embeddings)
        print_header "Averaging Train Embeddings per Emotion"
        python - <<'PY'
from utils.testing import average_emotion_embeddings_for_all

average_emotion_embeddings_for_all()
print("Averaging complete")
PY
        print_success "Emotion-wise averaging complete!"
        ;;

    avg-embeddings-speaker)
        SPEAKER="${2:-03}"
        print_header "Averaging Train Embeddings per Emotion for Speaker $SPEAKER"
        python - <<PY
from utils.testing import average_emotion_embeddings_for_speaker

average_emotion_embeddings_for_speaker("$SPEAKER")
print("Averaging complete for speaker $SPEAKER")
PY
        print_success "Emotion-wise averaging complete for speaker $SPEAKER!"
        ;;

    avg-test-embeddings)
        print_header "Averaging Test Embeddings by Base Id"
        python - <<'PY'
from utils.testing import average_test_embeddings_for_all

average_test_embeddings_for_all()
print("Test embedding averaging complete")
PY
        print_success "Test embedding averaging complete!"
        ;;

    avg-test-embeddings-speaker)
        SPEAKER="${2:-03}"
        print_header "Averaging Test Embeddings by Base Id for Speaker $SPEAKER"
        python - <<PY
from utils.testing import average_test_embeddings_for_speaker

average_test_embeddings_for_speaker("$SPEAKER")
print("Test embedding averaging complete for speaker $SPEAKER")
PY
        print_success "Test embedding averaging complete for speaker $SPEAKER!"
        ;;

    score-plda)
        print_header "PLDA Scoring for All Speakers"
        python - <<'PY'
from utils.testing import score_all_speakers_plda

score_all_speakers_plda()
print("PLDA scoring complete")
PY
        print_success "PLDA scoring complete!"
        ;;

    score-plda-speaker)
        SPEAKER="${2:-03}"
        print_header "PLDA Scoring for Speaker $SPEAKER"
        python - <<PY
from utils.testing import score_speaker_plda

score_speaker_plda("$SPEAKER")
print("PLDA scoring complete for speaker $SPEAKER")
PY
        print_success "PLDA scoring complete for speaker $SPEAKER!"
        ;;
    
    metadata)
        print_header "Extracting Metadata"
        python main.py --metadata
        print_success "Metadata extraction complete!"
        ;;
    
    segment)
        print_header "Segmenting Audio Files"
        python main.py --segment
        print_success "Audio segmentation complete!"
        ;;
    
    mfcc)
        print_header "Extracting MFCC Features"
        python main.py --mfcc
        print_success "MFCC extraction complete!"
        ;;
    
    create-mfcc-csv)
        print_header "Creating MFCC Features CSV"
        python utils/features_extraction/create_mfcc_csv.py
        print_success "MFCC CSV created!"
        ;;
    
    loso)
        print_header "Creating LOSO Splits"
        python main.py --loso
        print_success "LOSO splits created!"
        ;;
    
    splits)
        print_header "Creating Train/Val Splits"
        python main.py --splits
        print_success "Train/val splits created!"
        ;;
    
    verify)
        print_header "Verifying Installation"
        python verify_installation.py
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        usage
        exit 1
        ;;
esac
