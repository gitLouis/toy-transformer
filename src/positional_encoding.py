"""
Positional Encodings: inject sequence order information.

Since attention is permutation-invariant, we need to explicitly encode position.
Two approaches: sinusoidal (deterministic, good extrapolation) and learned (data-driven).
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class SinusoidalPositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encodings (Vaswani et al., 2017).
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Deterministic: no learned parameters
    - Extrapolates: can handle sequences longer than training
    - Relative positions: model can attend to "distance" via linear combinations
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Precompute positional encodings for efficiency
        positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]       # (1, d_model)
        
        # Compute angle rates: different frequencies for each dimension
        angle_rates = 1 / np.power(10000.0, (2 * (dims // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates  # (max_len, d_model)
        
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Store as constant tensor: (1, max_len, d_model)
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x, mask=None):
        """Add positional encoding to input embeddings."""
        seq_len = tf.shape(x)[1]
        output = x + self.pos_encoding[:, :seq_len, :]
        if mask is not None:
            output._keras_mask = mask
        return output


class LearnedPositionalEncoding(layers.Layer):
    """
    Learned positional embeddings: data-driven alternative to sinusoidal.
    
    Advantages: model can learn optimal positional representations
    Disadvantages: limited extrapolation beyond max_len seen during training
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        # Learnable positional embeddings: each position gets a d_model-dimensional vector
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.max_len, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x, mask=None):
        """Add learned positional embeddings to input."""
        seq_len = tf.shape(x)[1]
        output = x + self.pos_emb[tf.newaxis, :seq_len, :]
        if mask is not None:
            output._keras_mask = mask
        return output
