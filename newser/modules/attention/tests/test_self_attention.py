import unittest

import torch

from tell.modules.attention.self_attention import SelfAttention


class TestSelfAttention(unittest.TestCase):
    def test_self_attention_with_empty_incremental_state(self):
        attn = SelfAttention(4, 5, num_heads=1)

        X = torch.randn(7, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output = attn(X)
        # output.shape == [seq_len, batch_size, out_channels]

        output_incremental = attn(X, incremental_state={})
        self.assertAlmostEqual(output, output_incremental)

    def test_self_attention_using_incremental_state(self):
        attn = SelfAttention(4, 5, num_heads=1)

        X = torch.randn(7, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output_full = attn(X)
        # output.shape == [seq_len, batch_size, out_channels]

        incremental_state = {}
        for i in range(X.shape[0]):
            output_incr = attn(X[i:i+1], incremental_state)
            self.assertAlmostEqual(output_full[i:i+1], output_incr)
        # output_incr.shape == [1, batch_size, out_channels]

    def test_self_attention_using_multi_step_incremental_state(self):
        attn = SelfAttention(4, 5, num_heads=1)

        X = torch.randn(8, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output_full = attn(X)
        # output.shape == [seq_len, batch_size, out_channels]

        incremental_state = {}

        # Feed in first 3 steps
        output_incr = attn(X[:3], incremental_state)
        self.assertAlmostEqual(output_full[:3], output_incr)

        # Feed in next 1 step
        output_incr = attn(X[3:4], incremental_state)
        self.assertAlmostEqual(output_full[3:4], output_incr)

        # Feed in last 4 steps
        output_incr = attn(X[4:], incremental_state)
        self.assertAlmostEqual(output_full[4:], output_incr)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
