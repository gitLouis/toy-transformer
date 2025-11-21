"""
Core building blocks: Layer Normalization, Dropout, and Embeddings.

These components form the foundation of the Transformer architecture.
LayerNorm enables stable training, embeddings map discrete tokens to continuous space,
and dropout prevents overfitting.
"""

import tensorflow as tf
from tensorflow.keras import layers


class LayerNormalization(layers.Layer):
    """
    Layer Normalization: per-sample normalization across the feature dimension.
    
    Formula: LN(x) = γ * (x - μ) / √(σ² + ε) + β
    where μ, σ² are computed per sample across the last dimension.
    
    Why LayerNorm? Unlike BatchNorm, it's independent of batch statistics,
    making it ideal for variable-length sequences and stable training.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def build(self, input_shape):
        # Learnable affine parameters: scale (γ) and shift (β)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer="zeros", trainable=True)

    def call(self, x, mask=None):
        # Compute statistics across the feature dimension (last axis)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean((x - mean) ** 2, axis=-1, keepdims=True)
        
        # Normalize and apply affine transformation
        x_norm = (x - mean) / tf.sqrt(variance + self.eps)
        output = self.gamma * x_norm + self.beta
        
        # Preserve Keras mask for automatic masking propagation
        if mask is not None:
            output._keras_mask = mask
        return output


class ResidualDropout(layers.Layer):
    """
    Dropout applied to residual connections.
    
    Applied after attention/FFN outputs before adding to residual.
    Disabled during inference (training=False) for deterministic outputs.
    """
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def call(self, x, training=False, mask=None):
        # Only apply dropout during training
        output = tf.nn.dropout(x, rate=self.rate) if training and self.rate > 0.0 else x
        if mask is not None:
            output._keras_mask = mask
        return output


class EmbeddingLayer(layers.Layer):
    """
    Token embeddings with optional scaling.
    
    Maps discrete token IDs to dense vectors in d_model-dimensional space.
    Scaling by √d_model counteracts the reduction in variance from the dot product
    in attention (see "Attention is All You Need" Section 3.4).
    
    Supports automatic masking: padding tokens (0) generate masks for downstream layers.
    """
    def __init__(self, vocab_size, d_model, scale=True, mask_zero=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = scale
        self.mask_zero = mask_zero

    def build(self, input_shape):
        # Embedding matrix: each row is a d_model-dimensional vector for a token
        self.emb = self.add_weight(
            name="emb_matrix", 
            shape=(self.vocab_size, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True
        )

    def compute_mask(self, inputs, mask=None):
        """Generate mask from padding tokens (0) for Keras automatic masking."""
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

    def call(self, x):
        # Lookup embeddings: (batch, seq_len) -> (batch, seq_len, d_model)
        output = tf.nn.embedding_lookup(self.emb, x)
        
        # Scale by √d_model to maintain variance (Vaswani et al., 2017)
        if self.scale:
            output *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return output
