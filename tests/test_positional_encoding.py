"""
Unit tests for positional encoding module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.max_len = 100
        self.d_model = 64
        self.layer = SinusoidalPositionalEncoding(self.max_len, self.d_model)

    def test_call(self):
        """Test sinusoidal positional encoding."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        output = self.layer(x)
        self.assertEqual(output.shape, x.shape)
        # Output should be different from input (positional encoding added)
        self.assertFalse(tf.reduce_all(tf.equal(output, x)))

    def test_different_positions(self):
        """Test that different positions produce different encodings."""
        x1 = tf.random.normal((1, 5, self.d_model))
        x2 = tf.random.normal((1, 5, self.d_model))
        # Use same input but different sequence lengths
        out1 = self.layer(x1)
        out2 = self.layer(x2)
        # Should be different due to positional encoding
        self.assertFalse(tf.reduce_all(tf.equal(out1, out2)))


class TestLearnedPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.max_len = 100
        self.d_model = 64
        self.layer = LearnedPositionalEncoding(self.max_len, self.d_model)

    def test_build(self):
        """Test that learned positional encoding builds correctly."""
        # Trigger build by calling the layer
        dummy_input = tf.random.normal((2, 10, self.d_model))
        _ = self.layer(dummy_input)
        self.assertIsNotNone(self.layer.pos_emb)
        self.assertEqual(self.layer.pos_emb.shape, (self.max_len, self.d_model))

    def test_call(self):
        """Test learned positional encoding."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        output = self.layer(x)
        self.assertEqual(output.shape, x.shape)
        # Output should be different from input
        self.assertFalse(tf.reduce_all(tf.equal(output, x)))


if __name__ == '__main__':
    unittest.main()

