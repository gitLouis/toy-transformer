"""
Training utilities: masked loss function and training step.

The loss function ignores padding tokens, and the training step handles
gradient computation and optimization.
"""

import tensorflow as tf
from .masks import create_masks


def loss_function(real, pred):
    """
    Compute masked cross-entropy loss for sequence prediction.
    
    Only computes loss on non-padding tokens (where real != 0).
    This prevents the model from learning to predict padding tokens.
    
    Args:
        real: ground truth token IDs (batch, seq_len)
        pred: predicted logits (batch, seq_len, vocab_size)
    
    Returns:
        Scalar loss value (average over non-padding tokens)
    """
    # Create mask: 1.0 for real tokens, 0.0 for padding
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    
    # Compute per-token losses
    per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    
    # Apply mask and compute mean over non-padding tokens
    masked_loss = per_token_loss * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-9)


@tf.function
def train_step(model, src, tgt_inp, tgt_real, optimizer):
    """
    Single training step: forward pass, loss computation, backpropagation.
    
    Uses teacher forcing: during training, decoder receives ground truth target
    sequence (shifted by one position) rather than its own predictions.
    
    Args:
        model: Transformer model
        src: source sequences (batch, src_len)
        tgt_inp: target input (shifted ground truth) (batch, tgt_len)
        tgt_real: target ground truth (batch, tgt_len)
        optimizer: optimizer instance
    
    Returns:
        Scalar loss value
    """
    # Create masks for attention mechanisms
    enc_mask, combined_mask, dec_padding_mask = create_masks(src, tgt_inp)
    
    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(src, tgt_inp, 
                      enc_padding_mask=enc_mask, 
                      look_ahead_mask=combined_mask, 
                      dec_padding_mask=dec_padding_mask, 
                      training=True)
        
        # Compute loss
        loss = loss_function(tgt_real, logits)
    
    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
