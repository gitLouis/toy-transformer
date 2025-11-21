# Toy Transformer

A complete explicit encoder-decoder Transformer implementation in TensorFlow/Keras. This project demonstrates the core components of the Transformer architecture with explicit implementations of all layers, including embeddings, positional encodings, multi-head attention, encoder/decoder stacks, and masking mechanisms.

## Features

- **Explicit Layer Implementations**: 
  - Layer Normalization
  - Residual Dropout
  - Embedding Layer with scaling
  - Sinusoidal and Learned Positional Encodings
  - Scaled Dot-Product Attention
  - Multi-Head Attention with explicit QKV projections
  - Encoder and Decoder layers with residual connections

- **Masking Support**:
  - Padding masks for variable-length sequences
  - Look-ahead (causal) masks for autoregressive decoding
  - Combined decoder masks

- **Training & Inference**:
  - Masked loss function
  - Training step with gradient computation
  - Greedy decoding for inference

## Project Structure

```
toy-transformer/
├── src/
│   ├── __init__.py
│   ├── layers.py              # LayerNorm, Dropout, Embedding
│   ├── positional_encoding.py # Sinusoidal & Learned PE
│   ├── attention.py           # Scaled dot-product & Multi-head attention
│   ├── encoder.py             # Encoder layer & stack
│   ├── decoder.py             # Decoder layer & stack
│   ├── transformer.py         # Full Transformer model
│   ├── masks.py               # Mask creation utilities
│   ├── training.py            # Loss function & training step
│   ├── inference.py           # Greedy decoding
│   ├── data_preprocessing.py  # Text preprocessing utilities
│   ├── model_utils.py         # Model save/load utilities
│   └── gutenberg_downloader.py # Project Gutenberg book downloader
├── tests/
│   ├── test_layers.py
│   ├── test_positional_encoding.py
│   ├── test_attention.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_transformer.py
│   ├── test_masks.py
│   ├── test_training.py
│   └── test_inference.py
├── raw_data/                 # Example text files for training
│   ├── sample1.txt
│   ├── sample2.txt
│   └── sample3.txt
├── models/                   # Saved trained models (created after training)
├── train.py                  # Training script
├── download_gutenberg.py     # Gutenberg book downloader script (Python)
├── download_gutenberg_wget.sh # Gutenberg bulk downloader (wget, recommended)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── setup_venv.sh             # Virtual environment setup script
└── README.md                 # This file
```

## Installation

### Option 1: Using Virtual Environment (Recommended)

1. **Set up virtual environment:**
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies manually (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t toy-transformer .
   ```

2. **Run the container (runs tests by default):**
   ```bash
   docker run --rm toy-transformer
   ```

## Usage

### Installation

First, install the required dependencies:

```bash
# Using pip
pip install -r requirements.txt

# Or using the setup script (creates virtual environment)
bash setup_venv.sh
```

### Downloading Books from Project Gutenberg

There are two methods to download books:

#### Method 1: Using wget (Recommended for bulk downloads)

This uses Project Gutenberg's official robot/harvest endpoint, which is the recommended method for bulk downloads:

```bash
# Download all English text files
./download_gutenberg_wget.sh raw_data/gutenberg

# Or with default directory
./download_gutenberg_wget.sh
```

This method:
- Uses Project Gutenberg's official bulk download endpoint
- Respects their terms of service for automated downloads
- Downloads all English text files
- Waits 2 seconds between requests

#### Method 2: Using Python script (Top books only)

To download specific top books:

```bash
# Download top 5 books
python download_gutenberg.py --k 5

# Download top 10 books to a specific directory
python download_gutenberg.py --k 10 --dir raw_data/gutenberg
```

Or use the Python API:

```python
from src import download_top_k

# Download top 5 books
download_top_k(k=5, download_dir='raw_data/gutenberg')
```

**Note:** The Python script requires user approval for downloads > 3 books and waits 2 minutes between downloads to respect server resources.

### Training a Model

#### Quick Start: Full Training Cycle on Gutenberg Books

For a complete training cycle on Gutenberg books:

```bash
# Option 1: Use the automated script (recommended)
./train_gutenberg.sh

# Option 2: Manual steps
# 1. Download books
./download_gutenberg_wget.sh raw_data/gutenberg
# or
python download_gutenberg.py --k 10 --dir raw_data/gutenberg

# 2. Train the model
python train.py --data_dir raw_data/gutenberg --model_dir models
```

#### Training with Custom Options

```bash
# Train with custom parameters
python train.py \
    --data_dir raw_data/gutenberg \
    --model_dir models \
    --num_epochs 20 \
    --batch_size 8 \
    --max_len 100 \
    --d_model 128 \
    --num_heads 8 \
    --num_layers 4
```

#### Training on Custom Data

To train on your own text files:

```bash
# Place .txt files in a directory, then:
python train.py --data_dir my_text_files --model_dir my_models
```

#### What the Training Script Does

1. Loads text files from the specified directory
2. Preprocesses the data (tokenize and build vocabulary)
3. Creates a Transformer model
4. Trains the model on the preprocessed data
5. Saves the trained model to the specified directory

The saved model includes:
- Model weights (`transformer_model/`)
- Vocabulary mapping (`vocab.json`)
- Model configuration (`config.json`)

### Running Unit Tests

Run all tests:
```bash
python -m unittest discover -s tests -v
```

Run a specific test file:
```bash
python -m unittest tests.test_transformer -v
```

### Example: Creating and Using the Transformer

```python
import tensorflow as tf
from src import Transformer, train_step, greedy_decode, create_masks

# Create model
model = Transformer(
    num_layers=2,
    d_model=64,
    num_heads=4,
    ffn_dim=128,
    src_vocab_size=100,
    tgt_vocab_size=100,
    max_len=50,
    pos_encoding_type="sinusoidal",
    dropout_rate=0.1
)

# Create optimizer
optimizer = tf.keras.optimizers.Adam(1e-3)

# Training example
src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]], dtype=tf.int32)
tgt_inp = tf.constant([[1, 1, 2, 3, 4, 0, 0, 0, 0, 0]], dtype=tf.int32)
tgt_real = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]], dtype=tf.int32)

loss = train_step(model, src, tgt_inp, tgt_real, optimizer)
print(f"Loss: {loss.numpy()}")

# Inference example
src_seq = [1, 2, 3, 4, 5]
decoded = greedy_decode(model, src_seq, start_token=1, end_token=2, max_len=20)
print(f"Decoded: {decoded}")
```

## Configuration

The model can be configured with the following parameters:

- `num_layers`: Number of encoder/decoder layers (default: 2)
- `d_model`: Model dimension (default: 64)
- `num_heads`: Number of attention heads (default: 4)
- `ffn_dim`: Feed-forward network dimension (default: 128)
- `src_vocab_size`: Source vocabulary size
- `tgt_vocab_size`: Target vocabulary size
- `max_len`: Maximum sequence length
- `pos_encoding_type`: "sinusoidal" or "learned" (default: "sinusoidal")
- `dropout_rate`: Dropout rate (default: 0.1)

## Architecture Details

### Encoder
- Token embeddings + positional encoding
- Stack of encoder layers, each containing:
  - Multi-head self-attention
  - Residual connection + layer normalization
  - Position-wise feed-forward network
  - Residual connection + layer normalization

### Decoder
- Token embeddings + positional encoding
- Stack of decoder layers, each containing:
  - Masked multi-head self-attention (causal)
  - Residual connection + layer normalization
  - Multi-head cross-attention (over encoder outputs)
  - Residual connection + layer normalization
  - Position-wise feed-forward network
  - Residual connection + layer normalization
- Final linear projection to vocabulary logits

### Attention Mechanism
- Scaled dot-product attention: `softmax(QK^T / √d_k) V`
- Multi-head attention splits into parallel heads
- Masks applied to prevent attention to padding tokens and future positions

## Testing

The project includes comprehensive unit tests for all components:

- Layer normalization and dropout
- Embedding layers
- Positional encodings
- Attention mechanisms
- Encoder and decoder layers
- Full transformer model
- Masking utilities
- Training and inference functions

Run tests with:
```bash
python -m unittest discover -s tests -v
```

## Requirements

- Python 3.8+
- TensorFlow 2.13.0 or higher
- NumPy 1.24.0 or higher

## License

This is an educational implementation for learning purposes.

## Notes

- This is a toy implementation for educational purposes
- For production use, consider using established libraries like Hugging Face Transformers
- The model uses dummy data for demonstration
- Real applications would require proper tokenization and vocabulary building

