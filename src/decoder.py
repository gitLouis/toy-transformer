"""
Decoder: autoregressively generates target sequence from encoder outputs.

Each decoder layer contains three sub-layers:
1. Masked self-attention (causal: prevents attending to future tokens)
2. Cross-attention (queries from decoder, keys/values from encoder)
3. Position-wise feed-forward network

The masking ensures autoregressive property: each position can only attend to
previous positions in the target sequence.
"""

import tensorflow as tf
from tensorflow.keras import layers
from .layers import EmbeddingLayer, LayerNormalization, ResidualDropout
from .positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from .attention import MultiHeadAttention


class DecoderLayer(layers.Layer):
    """
    Single decoder layer: masked self-attention + cross-attention + FFN.
    
    Architecture:
        x -> masked self-attn -> add -> norm ->
        x -> cross-attn -> add -> norm ->
        x -> FFN -> add -> norm -> output
    """
    def __init__(self, d_model, num_heads, ffn_dim, attn_dropout=0.0, proj_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, proj_dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, proj_dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.ffn_dropout = ResidualDropout(ffn_dropout)

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        """
        Forward pass through decoder layer.
        
        Args:
            x: decoder input (batch, tgt_len, d_model)
            enc_output: encoder outputs (batch, src_len, d_model)
            look_ahead_mask: causal mask for self-attention
            padding_mask: mask for cross-attention (encoder padding)
            training: training mode flag
        """
        # Masked self-attention: causal masking prevents attending to future tokens
        attn1 = self.masked_self_attn(x, x, x, mask=look_ahead_mask, training=training)
        x = self.norm1(x + attn1)

        # Cross-attention: decoder queries attend to encoder outputs
        # This is where source information flows into the decoder
        attn2 = self.cross_attn(x, enc_output, enc_output, mask=padding_mask, training=training)
        x = self.norm2(x + attn2)

        # Position-wise feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        x = self.norm3(x + ffn_output)
        return x


class Decoder(layers.Layer):
    """
    Decoder stack: embeddings + positional encoding + N decoder layers + output projection.
    
    Generates target sequence autoregressively, attending to both previous target tokens
    (via masked self-attention) and source sequence (via cross-attention).
    """
    def __init__(self, num_layers, d_model, num_heads, ffn_dim, tgt_vocab_size, max_len,
                 pos_encoding_type="sinusoidal", dropout_rate=0.1):
        super().__init__()
        self.token_emb = EmbeddingLayer(tgt_vocab_size, d_model)
        
        if pos_encoding_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(max_len, d_model)
        else:
            self.pos_emb = LearnedPositionalEncoding(max_len, d_model)
        
        self.layers = [
            DecoderLayer(d_model, num_heads, ffn_dim,
                        attn_dropout=dropout_rate, proj_dropout=dropout_rate, ffn_dropout=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = ResidualDropout(dropout_rate)
        
        # Final projection to vocabulary: outputs logits over target vocabulary
        self.final_dense = layers.Dense(tgt_vocab_size)

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        """
        Decode target sequence from encoder outputs.
        
        Args:
            x: target token IDs (batch, tgt_len)
            enc_output: encoder outputs (batch, src_len, d_model)
            look_ahead_mask: causal mask for autoregressive generation
            padding_mask: mask for cross-attention
            training: training mode flag
        
        Returns:
            Logits over vocabulary (batch, tgt_len, tgt_vocab_size)
        """
        # Embed and add positional encoding
        x = self.token_emb(x)
        keras_mask = getattr(x, '_keras_mask', None)
        x = self.pos_emb(x, mask=keras_mask)
        x = self.dropout(x, training=training, mask=keras_mask)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask=look_ahead_mask, 
                     padding_mask=padding_mask, training=training)
        
        # Project to vocabulary logits
        logits = self.final_dense(x)
        return logits
