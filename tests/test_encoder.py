"""
Unit tests for encoder module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.encoder import EncoderLayer, Encoder


class TestEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.layer = EncoderLayer(self.d_model, self.num_heads, self.ffn_dim)

    def test_call(self):
        """Test encoder layer forward pass."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        output = self.layer(x, training=False)
        self.assertEqual(output.shape, x.shape)

    def test_call_with_mask(self):
        """Test encoder layer with mask."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        mask = tf.ones((batch_size, 1, 1, seq_len))
        output = self.layer(x, mask=mask, training=False)
        self.assertEqual(output.shape, x.shape)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.src_vocab_size = 100
        self.max_len = 50
        self.encoder = Encoder(
            self.num_layers, self.d_model, self.num_heads, self.ffn_dim,
            self.src_vocab_size, self.max_len
        )

    def test_call(self):
        """Test encoder forward pass."""
        batch_size = 2
        seq_len = 10
        src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        output = self.encoder(src, training=False)
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_call_with_mask(self):
        """Test encoder with padding mask."""
        src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        mask = tf.ones((2, 1, 1, 10))
        output = self.encoder(src, mask=mask, training=False)
        self.assertEqual(output.shape, (2, 10, self.d_model))


if __name__ == '__main__':
    unittest.main()

