"""
Unit tests for masks module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.masks import create_padding_mask, create_look_ahead_mask, create_decoder_mask, create_masks


class TestMasks(unittest.TestCase):
    def test_create_padding_mask(self):
        """Test padding mask creation."""
        seq = tf.constant([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=tf.int32)
        mask = create_padding_mask(seq)
        self.assertEqual(mask.shape, (2, 1, 1, 5))
        # Check that padding positions (0) are masked
        self.assertEqual(mask[0, 0, 0, 3].numpy(), 0.0)  # padding position
        self.assertEqual(mask[0, 0, 0, 0].numpy(), 1.0)  # real token

    def test_create_look_ahead_mask(self):
        """Test look-ahead mask creation."""
        size = 5
        mask = create_look_ahead_mask(size)
        self.assertEqual(mask.shape, (size, size))
        # Check lower triangular structure
        self.assertEqual(mask[0, 0].numpy(), 1.0)  # diagonal
        self.assertEqual(mask[0, 1].numpy(), 0.0)  # upper triangle
        self.assertEqual(mask[1, 0].numpy(), 1.0)  # lower triangle

    def test_create_decoder_mask(self):
        """Test combined decoder mask."""
        tgt = tf.constant([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=tf.int32)
        mask = create_decoder_mask(tgt)
        self.assertEqual(mask.shape, (2, 1, 5, 5))
        # Should be lower triangular (causal) and mask padding
        self.assertEqual(mask[0, 0, 0, 0].numpy(), 1.0)  # allowed
        self.assertEqual(mask[0, 0, 0, 1].numpy(), 0.0)  # future position

    def test_create_masks(self):
        """Test create_masks function."""
        src = tf.constant([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=tf.int32)
        tgt = tf.constant([[1, 2, 3, 0], [1, 2, 0, 0]], dtype=tf.int32)
        enc_mask, combined_mask, dec_padding_mask = create_masks(src, tgt)
        self.assertEqual(enc_mask.shape, (2, 1, 1, 5))
        self.assertEqual(combined_mask.shape, (2, 1, 4, 4))
        self.assertEqual(dec_padding_mask.shape, (2, 1, 1, 5))


if __name__ == '__main__':
    unittest.main()

