import unittest

import torch

from tell.modules.convolutions.base import ConvBCT, ConvTBC


class TestConvTBC(unittest.TestCase):
    def test_convtbc(self):
        conv_tbc = ConvTBC(4, 5, kernel_size=3, padding=1)
        # conv_tbc.weight_v.shape == [kernel_size, in_channels, out_channels]
        # conv_tbc.weight_g.shape == [1, 1, out_channels]
        conv_bct = ConvBCT(4, 5, kernel_size=3, padding=1)
        # conv_bct.weight_v.shape == [out_channels, in_channels, kernel_size]
        # conv_bct.weight_g.shape == [out_channels, 1, 1]

        conv_tbc.weight_g.data.copy_(conv_bct.weight_g.data.transpose(0, 2))
        conv_tbc.weight_v.data.copy_(conv_bct.weight_v.data.transpose(0, 2))
        conv_tbc.bias.data.copy_(conv_bct.bias.data)

        input_tbc = torch.randn(7, 2, 4, requires_grad=True)
        input_bct = input_tbc.data.transpose(0, 1).transpose(1, 2)
        input_bct.requires_grad = True

        output_tbc = conv_tbc(input_tbc)
        output_bct = conv_bct(input_bct)

        self.assertAlmostEqual(output_tbc.data.transpose(
            0, 1).transpose(1, 2), output_bct.data)

        grad_tbc = torch.randn(output_tbc.size())
        grad_bct = grad_tbc.transpose(0, 1).transpose(1, 2).contiguous()

        output_tbc.backward(grad_tbc)
        output_bct.backward(grad_bct)

        self.assertAlmostEqual(conv_tbc.weight_g.grad.data.transpose(0, 2),
                               conv_bct.weight_g.grad.data)
        self.assertAlmostEqual(conv_tbc.weight_v.grad.data.transpose(0, 2),
                               conv_bct.weight_v.grad.data)
        self.assertAlmostEqual(conv_tbc.bias.grad.data,
                               conv_bct.bias.grad.data)
        self.assertAlmostEqual(
            input_tbc.grad.data.transpose(0, 1).transpose(1, 2),
            input_bct.grad.data)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
