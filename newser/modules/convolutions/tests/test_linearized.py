import unittest

import torch

from newser.modules.convolutions.linearized import LinearizedConvolution


class TestLinearizedConvolution(unittest.TestCase):
    def test_linearized_with_empty_incremental_state(self):
        conv = LinearizedConvolution(4, 5, kernel_size=3, padding=2)
        # conv.weight_v.shape == [kernel_size, in_channels, out_channels]
        # conv.weight_g.shape == [1, 1, out_channels]

        X = torch.randn(7, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output = conv(X)
        output_incremental = conv(X, incremental_state={})
        self.assertAlmostEqual(output, output_incremental)

    def test_linearized_using_multi_step_incremental_state(self):
        conv = LinearizedConvolution(4, 5, kernel_size=3, padding=2)
        # conv.weight_v.shape == [kernel_size, in_channels, out_channels]
        # conv.weight_g.shape == [1, 1, out_channels]

        X = torch.randn(8, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output_full = conv(X)
        # output_full.shape == [seq_len, batch_size, out_channels]
        # import pudb
        # pudb.set_trace()
        incremental_state = {}

        # Feed in first 3 steps
        output_incr = conv(X[:3], incremental_state)
        self.assertAlmostEqual(output_full[:3], output_incr)

        # Feed in next 1 step
        output_incr = conv(X[3:4], incremental_state)
        self.assertAlmostEqual(output_full[3:4], output_incr)

        # Feed in final 4 steps
        output_incr = conv(X[4:], incremental_state)
        self.assertAlmostEqual(output_full[4:], output_incr)

    def test_linearized_using_incremental_state(self):
        conv = LinearizedConvolution(4, 5, kernel_size=3, padding=2)
        # conv.weight_v.shape == [kernel_size, in_channels, out_channels]
        # conv.weight_g.shape == [1, 1, out_channels]

        X = torch.randn(7, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output_full = conv(X)
        # output_full.shape == [seq_len, batch_size, out_channels]
        # import pudb
        # pudb.set_trace()
        incremental_state = {}
        for i in range(X.shape[0]):
            output_incr = conv(X[i:i+1], incremental_state)
            self.assertAlmostEqual(output_full[i:i+1], output_incr)
        # output_incr.shape == [1, batch_size, out_channels]

    def test_linearized_using_incremental_state_pad_1(self):
        conv = LinearizedConvolution(4, 5, kernel_size=3, padding=1)
        # conv.weight_v.shape == [kernel_size, in_channels, out_channels]
        # conv.weight_g.shape == [1, 1, out_channels]

        X = torch.randn(7, 2, 4, requires_grad=True)
        # X.shape == [seq_len, batch_size, in_channels]

        output_full = conv(X)
        # output_full.shape == [seq_len - 1, batch_size, out_channels]

        incremental_state = {}
        for i in range(X.shape[0]):
            output_incr = conv(X[i:i+1], incremental_state)
            # Because padding is 1, we ignore the first output. The first
            # output assumes that the padding is `kernel_size -1`.
            if i >= 1:
                self.assertAlmostEqual(output_full[i-1:i], output_incr)
        # output_incr.shape == [1, batch_size, out_channels]

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
