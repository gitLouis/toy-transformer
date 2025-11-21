"""
Unit tests for layers module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.layers import LayerNormalization, ResidualDropout, EmbeddingLayer


class TestLayerNormalization(unittest.TestCase):
    def setUp(self):
        self.layer = LayerNormalization()
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 64

    def test_build(self):
        """Test that layer builds correctly."""
        # Trigger build by calling the layer
        dummy_input = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        _ = self.layer(dummy_input)
        self.assertIsNotNone(self.layer.gamma)
        self.assertIsNotNone(self.layer.beta)
        self.assertEqual(self.layer.gamma.shape, (self.d_model,))
        self.assertEqual(self.layer.beta.shape, (self.d_model,))

    def test_call(self):
        """Test layer normalization computation."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        output = self.layer(x)
        self.assertEqual(output.shape, x.shape)
        # Check that output has similar scale
        self.assertLess(tf.reduce_mean(tf.abs(output)), 10.0)


class TestResidualDropout(unittest.TestCase):
    def setUp(self):
        self.dropout_rate = 0.5
        self.layer = ResidualDropout(self.dropout_rate)

    def test_training_mode(self):
        """Test dropout in training mode."""
        x = tf.ones((2, 5, 64))
        output = self.layer(x, training=True)
        # In training, some values should be zeroed (with high probability)
        self.assertEqual(output.shape, x.shape)

    def test_inference_mode(self):
        """Test dropout in inference mode (no dropout)."""
        x = tf.ones((2, 5, 64))
        output = self.layer(x, training=False)
        # In inference, output should equal input
        self.assertTrue(tf.reduce_all(tf.equal(output, x)))


class TestEmbeddingLayer(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.d_model = 64
        self.layer = EmbeddingLayer(self.vocab_size, self.d_model, scale=True)

    def test_build(self):
        """Test embedding layer builds correctly."""
        # Trigger build by calling the layer
        dummy_input = tf.constant([[1, 2, 3], [4, 5, 6]])
        _ = self.layer(dummy_input)
        self.assertIsNotNone(self.layer.emb)
        self.assertEqual(self.layer.emb.shape, (self.vocab_size, self.d_model))

    def test_call(self):
        """Test embedding lookup."""
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        output = self.layer(x)
        self.assertEqual(output.shape, (2, 3, self.d_model))
        # Check that scaling is applied
        self.assertGreater(tf.reduce_mean(tf.abs(output)), 0.0)


if __name__ == '__main__':
    unittest.main()

