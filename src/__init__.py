"""
Toy Transformer: Educational implementation of the Transformer architecture.

A complete, explicit implementation of the encoder-decoder Transformer from
"Attention is All You Need" (Vaswani et al., 2017), designed for educational purposes.

Key components:
- Scaled Dot-Product Attention and Multi-Head Attention
- Encoder and Decoder stacks with residual connections
- Positional encodings (sinusoidal and learned)
- Masking mechanisms for padding and causal attention
"""

from .transformer import Transformer
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadAttention, scaled_dot_product_attention
from .layers import LayerNormalization, ResidualDropout, EmbeddingLayer
from .positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from .masks import create_padding_mask, create_look_ahead_mask, create_decoder_mask, create_masks
from .training import loss_function, train_step
from .inference import greedy_decode
from .data_preprocessing import (
    read_text_files,
    simple_tokenize,
    build_vocab,
    encode_sequence,
    create_sequence_pairs,
    preprocess_data
)
from .model_utils import load_model

__all__ = [
    'Transformer',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'MultiHeadAttention',
    'scaled_dot_product_attention',
    'LayerNormalization',
    'ResidualDropout',
    'EmbeddingLayer',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'create_padding_mask',
    'create_look_ahead_mask',
    'create_decoder_mask',
    'create_masks',
    'loss_function',
    'train_step',
    'greedy_decode',
    'read_text_files',
    'simple_tokenize',
    'build_vocab',
    'encode_sequence',
    'create_sequence_pairs',
    'preprocess_data',
    'load_model',
]
