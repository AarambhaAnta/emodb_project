#!/bin/bash
# EmoDb Emotion Recognition Pipeline - Run Script
#
# 7-stage pipeline:
#   1. segment      - Segment audio files and create CSV
#   2. mfcc         - Extract MFCC features (396x40)
#   3. loso         - Create LOSO + 80/20 train/val splits
#   4. train        - Train ECAPA-TDNN model per speaker
#   5. embeddings   - Extract embeddings from trained models
#   6. train-plda   - Train PLDA model per speaker
#   7. test-plda    - Test PLDA model per speaker
#
# Usage: ./run.sh [command] [--speaker N]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

usage() {
    echo "EmoDb Emotion Recognition Pipeline"
    echo ""
    echo "Usage: ./run.sh <command> [--speaker N]"
    echo ""
    echo "Pipeline stages:"
    echo "  1  segment       Segment audio files and create CSV"
    echo "  2  mfcc          Extract MFCC features (396x40)"
    echo "  3  loso          Create LOSO + 80/20 train/val splits"
    echo "  4  train         Train ECAPA-TDNN model per speaker"
    echo "  5  embeddings    Extract embeddings from trained models"
    echo "  6  train-plda    Train PLDA model per speaker"
    echo "  7  test-plda     Test PLDA model"
    echo ""
    echo "Compound commands:"
    echo "  all              Run all 7 stages"
    echo "  preprocess       Stages 1-3 (segment + mfcc + loso)"
    echo "  pipeline         Stages 4-7 (train + embeddings + plda train + plda test)"
    echo ""
    echo "Options:"
    echo "  --speaker N      Run stage for specific speaker only (e.g., --speaker 03)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh all"
    echo "  ./run.sh train --speaker 03"
    echo "  ./run.sh embeddings --speaker 03"
    echo "  ./run.sh train-plda --speaker 03"
    echo "  ./run.sh test-plda --speaker 03"
}

# Parse speaker flag
SPEAKER_FLAG=""
COMMAND="$1"
shift || true
for arg in "$@"; do
    if [[ "$arg" == "--speaker" ]]; then
        shift
        SPEAKER_FLAG="--speaker $1"
        shift
    fi
done

case "$COMMAND" in
    # ----- individual stages -----
    segment)
        echo -e "${BLUE}Stage 1: Audio Segmentation${NC}"
        python main.py --segment
        ;;
    mfcc)
        echo -e "${BLUE}Stage 2: MFCC Feature Extraction${NC}"
        python main.py --mfcc
        ;;
    loso)
        echo -e "${BLUE}Stage 3: LOSO + Train/Val Splits${NC}"
        python main.py --loso
        ;;
    train)
        echo -e "${BLUE}Stage 4: ECAPA-TDNN Training${NC}"
        python main.py --train $SPEAKER_FLAG
        ;;
    embeddings)
        echo -e "${BLUE}Stage 5: Embedding Extraction${NC}"
        python main.py --embeddings $SPEAKER_FLAG
        ;;
    train-plda)
        echo -e "${BLUE}Stage 6: PLDA Training${NC}"
        python main.py --train-plda $SPEAKER_FLAG
        ;;
    test-plda)
        echo -e "${BLUE}Stage 7: PLDA Testing${NC}"
        python main.py --test-plda $SPEAKER_FLAG
        ;;

    # ----- compound commands -----
    preprocess)
        echo -e "${BLUE}Stages 1-3: Preprocessing${NC}"
        python main.py --segment --mfcc --loso
        ;;
    pipeline)
        echo -e "${BLUE}Stages 4-7: Training + PLDA${NC}"
        python main.py --train --embeddings --train-plda --test-plda $SPEAKER_FLAG
        ;;
    all)
        echo -e "${BLUE}Running all 7 stages${NC}"
        python main.py --all
        ;;

    # ----- help -----
    help|--help|-h|"")
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

echo -e "${GREEN}Done.${NC}"
