"""
Attention mechanism: the core innovation of Transformers.

Scaled Dot-Product Attention computes weighted combinations of values based on
query-key similarities. Multi-Head Attention runs multiple attention operations
in parallel, allowing the model to attend to different representation subspaces.
"""

import tensorflow as tf
from tensorflow.keras import layers
from .layers import ResidualDropout


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    The scaling factor 1/√d_k prevents dot products from growing large in magnitude,
    which would push softmax into regions with extremely small gradients.
    
    Args:
        Q: queries (batch, heads, q_len, head_dim)
        K: keys (batch, heads, k_len, head_dim)
        V: values (batch, heads, k_len, head_dim)
        mask: (batch, heads, q_len, k_len) with 1.0=keep, 0.0=mask
    
    Returns:
        Attention output (batch, heads, q_len, head_dim)
    """
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)
    
    # Compute attention scores: similarity between queries and keys
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
    
    # Apply mask: set masked positions to -∞ (becomes 0 after softmax)
    if mask is not None:
        scores += (1.0 - mask) * -1e9
    
    # Softmax over key dimension: attention weights sum to 1
    attention_weights = tf.nn.softmax(scores, axis=-1)
    
    # Weighted sum of values
    return tf.matmul(attention_weights, V)


class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention: parallel attention over different representation subspaces.
    
    Instead of one attention with d_model dimensions, we use h heads each with
    d_model/h dimensions. This allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Architecture:
    1. Linear projections: Q, K, V = X @ W_q, X @ W_k, X @ W_v
    2. Split into heads: (batch, seq_len, d_model) -> (batch, heads, seq_len, head_dim)
    3. Scaled dot-product attention per head
    4. Concatenate heads and project: (batch, seq_len, d_model)
    """
    def __init__(self, d_model, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

    def build(self, input_shape):
        # Learned projection matrices for Q, K, V, and output
        self.Wq = self.add_weight(name="Wq", shape=(self.d_model, self.d_model), initializer="glorot_uniform")
        self.Wk = self.add_weight(name="Wk", shape=(self.d_model, self.d_model), initializer="glorot_uniform")
        self.Wv = self.add_weight(name="Wv", shape=(self.d_model, self.d_model), initializer="glorot_uniform")
        self.Wo = self.add_weight(name="Wo", shape=(self.d_model, self.d_model), initializer="glorot_uniform")

        self._attn_drop = ResidualDropout(self.attn_dropout)
        self._proj_drop = ResidualDropout(self.proj_dropout)

    def _split_heads(self, x):
        """Reshape to split d_model into num_heads parallel subspaces."""
        # (batch, seq_len, d_model) -> (batch, heads, seq_len, head_dim)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """Concatenate heads back to d_model dimensions."""
        # (batch, heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        return tf.reshape(x, (batch_size, seq_len, self.d_model))

    def _process_mask(self, mask, q_len, k_len, batch_size):
        """
        Convert mask to attention format: (batch, heads, q_len, k_len).
        
        Handles various input formats:
        - Boolean Keras mask (batch, seq_len) -> float (batch, heads, q_len, k_len)
        - Float mask (batch, 1, 1, seq_len) -> broadcast to heads
        - Already formatted (batch, heads, q_len, k_len) -> use as-is
        """
        if mask is None:
            return None
            
        # Convert boolean to float if needed
        if mask.dtype == tf.bool:
            mask = tf.cast(mask, tf.float32)
        
        mask_rank = len(mask.shape)
        
        if mask_rank == 2:  # (batch, seq_len) - typically key sequence length
            mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)
            return tf.broadcast_to(mask, [batch_size, self.num_heads, q_len, k_len])
        elif mask_rank == 4:
            if mask.shape[1] == 1:  # (batch, 1, 1, seq_len) - broadcast to heads
                return tf.broadcast_to(mask, [batch_size, self.num_heads, q_len, k_len])
            else:  # Already (batch, heads, q_len, k_len)
                return tf.cast(mask, tf.float32)
        else:
            return tf.cast(mask, tf.float32)

    def call(self, query, key, value, mask=None, training=False):
        """
        Multi-head attention forward pass.
        
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: attention mask (various formats supported)
            training: whether in training mode
        
        Returns:
            (batch, seq_len, d_model)
        """
        # Get Keras automatic mask if available (for self-attention)
        query_mask = getattr(query, '_keras_mask', None)
        
        # Linear projections to Q, K, V
        Q = tf.tensordot(query, self.Wq, axes=[[2], [0]])  # (batch, q_len, d_model)
        K = tf.tensordot(key, self.Wk, axes=[[2], [0]])    # (batch, k_len, d_model)
        V = tf.tensordot(value, self.Wv, axes=[[2], [0]])  # (batch, k_len, d_model)

        # Split into multiple heads
        Qh = self._split_heads(Q)  # (batch, heads, q_len, head_dim)
        Kh = self._split_heads(K)  # (batch, heads, k_len, head_dim)
        Vh = self._split_heads(V)  # (batch, heads, k_len, head_dim)

        # Get sequence lengths (may differ for cross-attention)
        q_len = tf.shape(Qh)[2]
        k_len = tf.shape(Kh)[2]
        batch_size = tf.shape(Qh)[0]

        # Process mask: convert to attention format
        attention_mask = self._process_mask(mask, q_len, k_len, batch_size)
        
        # Use Keras mask if no manual mask provided (self-attention case)
        if attention_mask is None and query_mask is not None:
            query_mask = tf.cast(query_mask, tf.float32)
            mask_len = tf.shape(query_mask)[1]
            if mask_len == k_len:  # Only for self-attention (q_len == k_len)
                mask_tensor = query_mask[:, tf.newaxis, tf.newaxis, :]
                attention_mask = tf.broadcast_to(mask_tensor, [batch_size, self.num_heads, q_len, k_len])

        # Scaled dot-product attention per head
        attn_output = scaled_dot_product_attention(Qh, Kh, Vh, attention_mask)

        # Apply dropout to attention output
        attn_output = self._attn_drop(attn_output, training=training)

        # Concatenate heads and project
        combined = self._combine_heads(attn_output)  # (batch, q_len, d_model)
        output = tf.tensordot(combined, self.Wo, axes=[[2], [0]])  # (batch, q_len, d_model)
        output = self._proj_drop(output, training=training)
        
        # Propagate Keras mask for automatic masking
        if query_mask is not None:
            output._keras_mask = query_mask
        return output
