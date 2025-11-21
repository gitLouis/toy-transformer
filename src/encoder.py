"""
Encoder: processes source sequence into contextualized representations.

The encoder stack consists of N identical layers, each containing:
1. Multi-head self-attention (allows tokens to attend to all positions)
2. Position-wise feed-forward network (applied independently per position)
Both sub-layers use residual connections and layer normalization (post-norm architecture).
"""

import tensorflow as tf
from tensorflow.keras import layers
from .layers import EmbeddingLayer, LayerNormalization, ResidualDropout
from .positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from .attention import MultiHeadAttention


class EncoderLayer(layers.Layer):
    """
    Single encoder layer: self-attention + feed-forward with residual connections.
    
    Architecture (post-norm):
        x -> self-attention -> add -> layer_norm -> FFN -> add -> layer_norm -> output
    
    Post-norm vs Pre-norm: Post-norm applies normalization after residual addition.
    Pre-norm (x -> norm -> attention -> add) can sometimes train more stably.
    """
    def __init__(self, d_model, num_heads, ffn_dim, attn_dropout=0.0, proj_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, proj_dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        # Position-wise FFN: two linear transformations with ReLU activation
        # Applied independently to each position (hence "position-wise")
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.ffn_dropout = ResidualDropout(ffn_dropout)

    def call(self, x, mask=None, training=False):
        """Forward pass through encoder layer."""
        # Self-attention: each position attends to all positions (including itself)
        attn_output = self.self_attn(x, x, x, mask=mask, training=training)
        x = self.norm1(x + attn_output)  # Residual connection + layer norm

        # Position-wise feed-forward: MLP applied independently per position
        ffn_output = self.ffn(x)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        x = self.norm2(x + ffn_output)  # Residual connection + layer norm
        return x


class Encoder(layers.Layer):
    """
    Encoder stack: token embeddings + positional encoding + N encoder layers.
    
    Processes source sequence into rich contextualized representations that
    the decoder can attend to during cross-attention.
    """
    def __init__(self, num_layers, d_model, num_heads, ffn_dim, src_vocab_size, max_len,
                 pos_encoding_type="sinusoidal", dropout_rate=0.1):
        super().__init__()
        self.token_emb = EmbeddingLayer(src_vocab_size, d_model)
        
        if pos_encoding_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(max_len, d_model)
        else:
            self.pos_emb = LearnedPositionalEncoding(max_len, d_model)
        
        # Stack of N identical encoder layers
        self.layers = [
            EncoderLayer(d_model, num_heads, ffn_dim,
                        attn_dropout=dropout_rate, proj_dropout=dropout_rate, ffn_dropout=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = ResidualDropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        """
        Encode source sequence.
        
        Args:
            x: token IDs (batch, src_len)
            mask: padding mask for attention
            training: training mode flag
        
        Returns:
            Encoded representations (batch, src_len, d_model)
        """
        # Embed tokens and add positional information
        x = self.token_emb(x)
        keras_mask = getattr(x, '_keras_mask', None)
        x = self.pos_emb(x, mask=keras_mask)
        x = self.dropout(x, training=training, mask=keras_mask)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask, training=training)
        return x
