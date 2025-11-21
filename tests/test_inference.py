"""
Unit tests for inference module
"""

import unittest
import tensorflow as tf
import numpy as np
from src.inference import greedy_decode
from src.transformer import Transformer


class TestInference(unittest.TestCase):
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

    def test_greedy_decode(self):
        """Test greedy decoding."""
        src_seq = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0], dtype=np.int32)
        decoded = greedy_decode(self.model, src_seq, start_token=1, end_token=2, max_len=20)
        self.assertIsInstance(decoded, list)
        self.assertGreater(len(decoded), 0)
        # First token should be start token
        self.assertEqual(decoded[0], 1)


if __name__ == '__main__':
    unittest.main()

