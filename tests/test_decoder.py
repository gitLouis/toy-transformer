"""
Unit tests for decoder module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.decoder import DecoderLayer, Decoder


class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.layer = DecoderLayer(self.d_model, self.num_heads, self.ffn_dim)

    def test_call(self):
        """Test decoder layer forward pass."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        enc_output = tf.random.normal((batch_size, seq_len, self.d_model))
        output = self.layer(x, enc_output, training=False)
        self.assertEqual(output.shape, x.shape)

    def test_call_with_masks(self):
        """Test decoder layer with masks."""
        batch_size = 2
        seq_len = 10
        x = tf.random.normal((batch_size, seq_len, self.d_model))
        enc_output = tf.random.normal((batch_size, seq_len, self.d_model))
        look_ahead_mask = tf.ones((batch_size, 1, seq_len, seq_len))
        padding_mask = tf.ones((batch_size, 1, 1, seq_len))
        output = self.layer(x, enc_output, look_ahead_mask=look_ahead_mask, 
                           padding_mask=padding_mask, training=False)
        self.assertEqual(output.shape, x.shape)


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.tgt_vocab_size = 100
        self.max_len = 50
        self.decoder = Decoder(
            self.num_layers, self.d_model, self.num_heads, self.ffn_dim,
            self.tgt_vocab_size, self.max_len
        )

    def test_call(self):
        """Test decoder forward pass."""
        batch_size = 2
        seq_len = 10
        tgt = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        enc_output = tf.random.normal((batch_size, seq_len, self.d_model))
        logits = self.decoder(tgt, enc_output, training=False)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.tgt_vocab_size))

    def test_call_with_masks(self):
        """Test decoder with masks."""
        tgt = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        enc_output = tf.random.normal((2, 10, self.d_model))
        look_ahead_mask = tf.ones((2, 1, 10, 10))
        padding_mask = tf.ones((2, 1, 1, 10))
        logits = self.decoder(tgt, enc_output, look_ahead_mask=look_ahead_mask,
                             padding_mask=padding_mask, training=False)
        self.assertEqual(logits.shape, (2, 10, self.tgt_vocab_size))


if __name__ == '__main__':
    unittest.main()

