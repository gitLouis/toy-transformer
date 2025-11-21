"""
Unit tests for training module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.training import loss_function, train_step
from src.transformer import Transformer


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.model = Transformer(
            num_layers=2,
            d_model=64,
            num_heads=4,
            ffn_dim=128,
            src_vocab_size=100,
            tgt_vocab_size=100,
            max_len=50
        )
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def test_loss_function(self):
        """Test loss function computation."""
        real = tf.constant([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=tf.int32)
        pred = tf.random.normal((2, 5, 100))
        loss = loss_function(real, pred)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreater(loss.numpy(), 0.0)

    def test_train_step(self):
        """Test training step."""
        src = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                          [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        tgt_inp = tf.constant([[1, 1, 2, 3, 0, 0, 0, 0],
                              [1, 1, 2, 0, 0, 0, 0, 0]], dtype=tf.int32)
        tgt_real = tf.constant([[1, 2, 3, 4, 0, 0, 0, 0],
                               [1, 2, 3, 0, 0, 0, 0, 0]], dtype=tf.int32)
        loss = train_step(self.model, src, tgt_inp, tgt_real, self.optimizer)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreater(loss.numpy(), 0.0)


if __name__ == '__main__':
    unittest.main()

