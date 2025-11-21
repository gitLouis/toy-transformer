"""
Unit tests for attention module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 5
        self.head_dim = 16

    def test_attention_shape(self):
        """Test attention output shape."""
        Q = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        K = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        V = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        output = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))

    def test_attention_with_mask(self):
        """Test attention with masking."""
        Q = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        K = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        V = tf.random.normal((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        mask = tf.ones((self.batch_size, 1, self.seq_len, self.seq_len))
        output = scaled_dot_product_attention(Q, K, V, mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10
        self.layer = MultiHeadAttention(self.d_model, self.num_heads)

    def test_build(self):
        """Test that multi-head attention builds correctly."""
        # Trigger build by calling the layer
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        _ = self.layer(query, key, value, training=False)
        # Check that weights were created
        self.assertIsNotNone(self.layer.Wq)
        self.assertIsNotNone(self.layer.Wk)
        self.assertIsNotNone(self.layer.Wv)
        self.assertIsNotNone(self.layer.Wo)
        self.assertEqual(self.layer.Wq.shape, (self.d_model, self.d_model))
        self.assertEqual(self.layer.Wk.shape, (self.d_model, self.d_model))
        self.assertEqual(self.layer.Wv.shape, (self.d_model, self.d_model))
        self.assertEqual(self.layer.Wo.shape, (self.d_model, self.d_model))

    def test_call(self):
        """Test multi-head attention forward pass."""
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        output = self.layer(query, key, value, training=False)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_call_with_mask(self):
        """Test multi-head attention with mask."""
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        mask = tf.ones((self.batch_size, 1, self.seq_len, self.seq_len))
        output = self.layer(query, key, value, mask=mask, training=False)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_invalid_num_heads(self):
        """Test that invalid num_heads raises error."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=5)  # 64 not divisible by 5


if __name__ == '__main__':
    unittest.main()

