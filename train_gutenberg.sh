#!/bin/bash
#
# Full training cycle for Gutenberg books.
#
# This script:
# 1. Downloads Gutenberg books (if not already downloaded)
# 2. Preprocesses the data
# 3. Trains the Transformer model
# 4. Saves the trained model
#
# Usage:
#     ./train_gutenberg.sh [gutenberg_dir] [output_model_dir]
#
# Arguments:
#     gutenberg_dir: Directory containing Gutenberg .txt files (default: raw_data/gutenberg)
#     output_model_dir: Directory to save trained model (default: models)
#

set -e  # Exit on error

# Default directories
GUTENBERG_DIR="${1:-raw_data/gutenberg}"
MODEL_DIR="${2:-models}"

echo "============================================================"
echo "Full Training Cycle: Gutenberg Books"
echo "============================================================"
echo "Gutenberg data directory: $GUTENBERG_DIR"
echo "Model output directory: $MODEL_DIR"
echo ""

# Step 1: Check if Gutenberg books exist
echo "Step 1: Checking for Gutenberg books..."
if [ ! -d "$GUTENBERG_DIR" ] || [ -z "$(find "$GUTENBERG_DIR" -maxdepth 1 -name "*.txt" -type f 2>/dev/null)" ]; then
    echo "No Gutenberg books found in $GUTENBERG_DIR"
    echo ""
    echo "Would you like to download books now? (yes/no)"
    read -r response
    if [[ "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo ""
        echo "Downloading Gutenberg books..."
        echo "You can use either:"
        echo "  1. ./download_gutenberg_wget.sh $GUTENBERG_DIR  (bulk download, recommended)"
        echo "  2. python download_gutenberg.py --k 10 --dir $GUTENBERG_DIR  (top books only)"
        echo ""
        echo "Please run one of the above commands, then run this script again."
        exit 1
    else
        echo "Exiting. Please download books first."
        exit 1
    fi
else
    BOOK_COUNT=$(find "$GUTENBERG_DIR" -maxdepth 1 -name "*.txt" -type f | wc -l | tr -d ' ')
    echo "Found $BOOK_COUNT .txt files in $GUTENBERG_DIR"
fi

echo ""
echo "Step 2: Training model on Gutenberg books..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run training script
python train.py --data_dir "$GUTENBERG_DIR" --model_dir "$MODEL_DIR"

echo ""
echo "============================================================"
echo "Training cycle complete!"
echo "============================================================"
echo "Model saved to: $MODEL_DIR"
echo ""
echo "To use the trained model:"
echo "  from src.model_utils import load_model"
echo "  model, vocab, config = load_model('$MODEL_DIR')"

