"""
Model utilities: save and load trained models.
"""

import os
import json
import tensorflow as tf
from .transformer import Transformer


def load_model(model_dir: str) -> tuple:
    """
    Load a saved Transformer model and its metadata.
    
    Args:
        model_dir: directory containing saved model files
    
    Returns:
        Tuple of (model, vocab, config)
    """
    # Load configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir, 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Recreate model architecture
    model = Transformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        max_len=config['max_len'],
        pos_encoding_type=config['pos_encoding_type'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load weights
    weights_path = os.path.join(model_dir, 'transformer_model')
    model.load_weights(weights_path)
    
    return model, vocab, config

