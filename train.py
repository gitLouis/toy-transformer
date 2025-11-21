"""
Training script: preprocess data and train a Transformer model.

This script:
1. Loads and preprocesses text files from raw_data/
2. Creates a Transformer model
3. Trains the model on the preprocessed data
4. Saves the trained model to models/
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Tuple
from src.data_preprocessing import preprocess_data
from src.transformer import Transformer
from src.training import train_step
from src.masks import create_masks


def create_model(vocab_size: int, max_len: int, config: dict) -> Transformer:
    """
    Create and initialize Transformer model.
    
    Args:
        vocab_size: vocabulary size
        max_len: maximum sequence length
        config: model configuration dictionary
    
    Returns:
        Initialized Transformer model
    """
    model = Transformer(
        num_layers=config.get('num_layers', 2),
        d_model=config.get('d_model', 64),
        num_heads=config.get('num_heads', 4),
        ffn_dim=config.get('ffn_dim', 128),
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_len=max_len,
        pos_encoding_type=config.get('pos_encoding_type', 'sinusoidal'),
        dropout_rate=config.get('dropout_rate', 0.1)
    )
    return model


def prepare_target_sequences(target_seqs: np.ndarray, vocab: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare target sequences for teacher forcing.
    
    For teacher forcing:
    - tgt_inp: target sequence with START token prepended (shifted right)
    - tgt_real: target sequence (ground truth)
    
    Args:
        target_seqs: target sequences (batch, seq_len)
        vocab: vocabulary dictionary
    
    Returns:
        Tuple of (tgt_inp, tgt_real)
    """
    batch_size, seq_len = target_seqs.shape
    start_token = vocab['<START>']
    
    # Input: prepend START token, remove last token
    tgt_inp = np.concatenate([
        np.full((batch_size, 1), start_token, dtype=np.int32),
        target_seqs[:, :-1]
    ], axis=1)
    
    # Real: target sequence as-is
    tgt_real = target_seqs
    
    return tgt_inp, tgt_real


def train_model(model: Transformer, src_seqs: np.ndarray, tgt_seqs: np.ndarray,
               vocab: dict, config: dict):
    """
    Train the Transformer model.
    
    Args:
        model: Transformer model
        src_seqs: source sequences (batch, src_len)
        tgt_seqs: target sequences (batch, tgt_len)
        vocab: vocabulary dictionary
        config: training configuration
    """
    # Prepare target sequences for teacher forcing
    tgt_inp, tgt_real = prepare_target_sequences(tgt_seqs, vocab)
    
    # Convert to tensors
    src_tensor = tf.constant(src_seqs)
    tgt_inp_tensor = tf.constant(tgt_inp)
    tgt_real_tensor = tf.constant(tgt_real)
    
    # Create optimizer
    learning_rate = config.get('learning_rate', 1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Training loop
    num_epochs = config.get('num_epochs', 10)
    batch_size = config.get('batch_size', 4)
    num_samples = len(src_seqs)
    
    print(f"Training on {num_samples} samples for {num_epochs} epochs...")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sequence length: {src_seqs.shape[1]}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            src_batch = src_tensor[i:end_idx]
            tgt_inp_batch = tgt_inp_tensor[i:end_idx]
            tgt_real_batch = tgt_real_tensor[i:end_idx]
            
            # Training step
            loss = train_step(model, src_batch, tgt_inp_batch, tgt_real_batch, optimizer)
            epoch_losses.append(loss.numpy())
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    print("-" * 50)
    print("Training completed!")


def save_model(model: Transformer, vocab: dict, save_dir: str, config: dict):
    """
    Save trained model and metadata.
    
    Args:
        model: trained Transformer model
        vocab: vocabulary dictionary
        save_dir: directory to save model
        config: model configuration
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights (Keras 3 requires .weights.h5 extension)
    model_path = os.path.join(save_dir, 'transformer_model.weights.h5')
    model.save_weights(model_path)
    print(f"Model weights saved to {model_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(save_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Save configuration (include vocab_size and max_len for model recreation)
    save_config = config.copy()
    save_config['vocab_size'] = len(vocab)
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2)
    print(f"Configuration saved to {config_path}")


def main():
    """Main training pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Transformer on text data')
    parser.add_argument('--data_dir', type=str, default='raw_data',
                       help='Directory containing text files (default: raw_data)')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save trained model (default: models)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length (default: 50)')
    parser.add_argument('--d_model', type=int, default=64,
                       help='Model dimension (default: 64)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of encoder/decoder layers (default: 2)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'ffn_dim': args.d_model * 2,  # Typically 2-4x d_model
        'pos_encoding_type': 'sinusoidal',
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'tokenization_level': 'word',
        'min_token_freq': 1
    }
    
    # Paths
    data_dir = args.data_dir
    save_dir = args.model_dir
    
    print("=" * 50)
    print("Transformer Training Pipeline")
    print("=" * 50)
    
    # Preprocess data
    print("\n1. Preprocessing data...")
    src_seqs, tgt_seqs, vocab = preprocess_data(
        data_dir=data_dir,
        max_len=config['max_len'],
        level=config['tokenization_level'],
        min_freq=config['min_token_freq']
    )
    
    if len(src_seqs) == 0:
        print("Error: No valid training sequences found!")
        return
    
    print(f"   Processed {len(src_seqs)} sequences")
    print(f"   Vocabulary size: {len(vocab)}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_model(len(vocab), config['max_len'], config)
    
    # Build the model by calling it with sample inputs
    # This is necessary before counting parameters
    sample_src = tf.zeros((1, config['max_len']), dtype=tf.int32)
    sample_tgt = tf.zeros((1, config['max_len']), dtype=tf.int32)
    _ = model(sample_src, sample_tgt, training=False)
    
    print(f"   Model created with {model.count_params():,} parameters")
    
    # Train model
    print("\n3. Training model...")
    train_model(model, src_seqs, tgt_seqs, vocab, config)
    
    # Save model
    print("\n4. Saving model...")
    save_model(model, vocab, save_dir, config)
    
    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()

