"""
Unit tests for transformer model
"""

import unittest
import tensorflow as tf
import numpy as np
from src.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.max_len = 50
        self.model = Transformer(
            self.num_layers, self.d_model, self.num_heads, self.ffn_dim,
            self.src_vocab_size, self.tgt_vocab_size, self.max_len
        )

    def test_call(self):
        """Test transformer forward pass."""
        batch_size = 2
        src_len = 10
        tgt_len = 8
        src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        tgt = tf.constant([[1, 2, 3, 4, 0, 0, 0, 0],
                          [1, 2, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        logits = self.model(src, tgt, training=False)
        self.assertEqual(logits.shape, (batch_size, tgt_len, self.tgt_vocab_size))

    def test_call_with_masks(self):
        """Test transformer with masks."""
        src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        tgt = tf.constant([[1, 2, 3, 4, 0, 0, 0, 0],
                          [1, 2, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        enc_mask = tf.ones((2, 1, 1, 10))
        look_ahead_mask = tf.ones((2, 1, 8, 8))
        dec_padding_mask = tf.ones((2, 1, 1, 10))
        logits = self.model(src, tgt, enc_padding_mask=enc_mask,
                           look_ahead_mask=look_ahead_mask,
                           dec_padding_mask=dec_padding_mask, training=False)
        self.assertEqual(logits.shape, (2, 8, self.tgt_vocab_size))


if __name__ == '__main__':
    unittest.main()

