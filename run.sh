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
    echo "  train            - Train all models"
    echo "  train-speaker N  - Train specific speaker (e.g., train-speaker 03)"
    echo "  metadata         - Extract metadata only"
    echo "  segment          - Segment audio only"
    echo "  mfcc             - Extract MFCC features only"
    echo "  loso             - Create LOSO splits only"
    echo "  splits           - Create train/val splits only"
    echo "  verify           - Verify installation"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh all                # Run complete pipeline"
    echo "  ./run.sh preprocess         # Preprocessing only"
    echo "  ./run.sh train-speaker 03   # Train speaker 03"
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
        print_header "Training Speaker $SPEAKER"
        python main.py --train --speaker "$SPEAKER"
        print_success "Training complete for speaker $SPEAKER!"
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
