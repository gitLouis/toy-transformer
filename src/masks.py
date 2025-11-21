"""
Masking utilities for Transformer attention mechanisms.

Masks control which positions can attend to which other positions:
- Padding masks: prevent attention to padding tokens
- Look-ahead masks: enforce causal/autoregressive property in decoder
"""

import tensorflow as tf


def create_padding_mask(seq):
    """
    Create padding mask: 1.0 for real tokens, 0.0 for padding (token ID 0).
    
    Returns shape (batch, 1, 1, seq_len) suitable for broadcasting in attention.
    """
    mask = tf.cast(tf.not_equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create causal mask: lower triangular matrix.
    
    Prevents positions from attending to future positions, enforcing
    the autoregressive property in decoder self-attention.
    
    Returns (size, size) with 1.0 in lower triangle (including diagonal), 0.0 above.
    """
    return tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_decoder_mask(tgt):
    """
    Combined decoder mask: padding mask + look-ahead mask.
    
    Ensures decoder can neither attend to padding tokens nor future positions.
    Returns (batch, 1, tgt_len, tgt_len).
    """
    padding_mask = create_padding_mask(tgt)  # (batch, 1, 1, tgt_len)
    seq_len = tf.shape(tgt)[1]
    look_ahead = create_look_ahead_mask(seq_len)  # (tgt_len, tgt_len)
    look_ahead = look_ahead[tf.newaxis, tf.newaxis, :, :]  # (1, 1, tgt_len, tgt_len)
    
    # Element-wise multiply: both conditions must be satisfied
    return padding_mask * look_ahead


def create_masks(src, tgt):
    """
    Create all masks needed for Transformer forward pass.
    
    Returns:
        enc_padding_mask: for encoder self-attention (batch, 1, 1, src_len)
        combined_mask: for decoder self-attention (batch, 1, tgt_len, tgt_len)
        dec_padding_mask: for decoder cross-attention (batch, 1, 1, src_len)
    """
    enc_padding_mask = create_padding_mask(src)
    dec_padding_mask = create_padding_mask(src)  # Cross-attention uses encoder sequence length
    combined_mask = create_decoder_mask(tgt)
    return enc_padding_mask, combined_mask, dec_padding_mask
