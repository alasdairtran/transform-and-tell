import unittest

import torch

from tell.modules.token_embedders.positional import make_positions


class TestEmbeddings(unittest.TestCase):
    def test_make_positions(self):
        pad = 1

        left_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [1, 9, 9, 9, 9],
            [1, 1, 1, 9, 9],
        ])
        left_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 2, 3],
        ])

        right_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 1],
            [9, 9, 1, 1, 1],
        ])
        right_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [2, 3, 4, 5, 1],
            [2, 3, 1, 1, 1],
        ])

        print(left_pad_output)
        print(make_positions(left_pad_input, pad, left_pad=True))
        self.assertAlmostEqual(left_pad_output,
                               make_positions(left_pad_input, pad, left_pad=True))

        self.assertAlmostEqual(right_pad_output,
                               make_positions(right_pad_input, pad, left_pad=False))

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max().item(), 1e-4)
