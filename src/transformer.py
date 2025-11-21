"""
Complete Transformer model: encoder-decoder architecture.

The Transformer combines encoder and decoder stacks to perform sequence-to-sequence
tasks. The encoder processes the source sequence, and the decoder generates the
target sequence while attending to encoder outputs.
"""

import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


class Transformer(tf.keras.Model):
    """
    Transformer: encoder-decoder architecture for sequence-to-sequence tasks.
    
    The model processes source and target sequences in parallel during training
    (teacher forcing) and autoregressively during inference.
    """
    def __init__(self, num_layers, d_model, num_heads, ffn_dim, src_vocab_size, tgt_vocab_size, max_len,
                 pos_encoding_type="sinusoidal", dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, ffn_dim, src_vocab_size, max_len, 
                               pos_encoding_type, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, ffn_dim, tgt_vocab_size, max_len, 
                              pos_encoding_type, dropout_rate)

    def call(self, src, tgt, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None, training=False):
        """
        Forward pass: encode source, decode target.
        
        Args:
            src: source token IDs (batch, src_len)
            tgt: target token IDs (batch, tgt_len)
            enc_padding_mask: mask for encoder self-attention
            look_ahead_mask: causal mask for decoder self-attention
            dec_padding_mask: mask for decoder cross-attention
            training: training mode flag
        
        Returns:
            Logits over target vocabulary (batch, tgt_len, tgt_vocab_size)
        """
        # Encode source sequence
        enc_output = self.encoder(src, mask=enc_padding_mask, training=training)
        
        # Decode target sequence (attending to encoder outputs)
        logits = self.decoder(tgt, enc_output, look_ahead_mask=look_ahead_mask, 
                             padding_mask=dec_padding_mask, training=training)
        return logits
