
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _single


class ConvTBC(nn.Module):
    """1D convolution over an input of shape [seq_len, batch_size, in_channels].

    The implementation uses GEMM to perform the convolution. This
    implementation is faster than cuDNN for small kernel sizes. It is the same
    as torch.nn.Conv1d, except it accepts the kernel size as the first
    dimension (instead of the last) in the weight matrix. The kernel size is
    the time dimension, the number of words in the window. The in_channels is
    the input hidden size and the out_channels is the output hidden size.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0,
                 padding=0, weight_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.weight_norm = weight_norm
        self.reset_parameters()

    def reset_parameters(self):
        # See A.3. in Gehring et al. (2017) for the justification of the
        # constant 4: https://arxiv.org/pdf/1705.03122.
        std = math.sqrt((4 * (1.0 - self.dropout)) /
                        (self.kernel_size[0] * self.in_channels))
        self.weight.data.normal_(mean=0, std=std)
        self.bias.data.fill_(0)
        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. The norm is computed
        # independently per output channel (dim 2).
        if self.weight_norm:
            nn.utils.weight_norm(self, dim=2)

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvBCT(nn.Conv1d):
    """1D convolution over an input of shape [batch_size, in_channels, seq_len].

    A wrapper of Conv1d with Gehring initialization and weight normalization.
    """

    def reset_parameters(self):
        # See A.3. in Gehring et al. (2017) for the justification of the
        # constant 4: https://arxiv.org/pdf/1705.03122.
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = math.sqrt(4 / n)
        self.weight.data.normal_(mean=0, std=std)
        self.bias.data.fill_(0)
        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. The norm is computed
        # independently per output channel (dim 0).
        nn.utils.weight_norm(self, dim=0)
