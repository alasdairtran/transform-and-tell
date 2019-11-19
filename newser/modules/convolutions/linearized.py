# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F

from tell.utils import get_incremental_state, set_incremental_state

from .base import ConvTBC


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that in the original Fairseq implementation, the input order changes
    from training (time dimension first) to inference (batch dimension first).
    In this new implementation, for consistency, LinearizedConvolution only
    accepts inputs with time dimension first.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def forward(self, X, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then X is
                expected to contain a single frame. If the X order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel
        """

        # We shall take care of three cases. If incremental state is not
        # supplied at well, we fall back to the standard ConvTBC
        # implementation, which does a full convolution over all words in a
        # sequence.
        if incremental_state is None:
            output = super().forward(X)
            # output.shape == [seq_len + padding, batch_size, out_channels]
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                # remove future timesteps added by padding
                output = output[:-self.padding[0], :, :]
            return output

        # Otherwise we're in incremental mode. If we're only provided with
        # one step, we'll perform the convolution using a simple linear layer.
        # We assume that the first dimension is time.
        elif X.shape[0] == 1:
            output = self._forward_one_step(X, incremental_state)
            return output

        else:
            output = self._forward_multiple_steps(X, incremental_state)
            return output

    def _forward_one_step(self, X, incremental_state):
        # X.shape == [seq_len, batch_size, in_channels]

        kernel_width = self.kernel_size[0]
        # weight.shape == [in_channels, kernel_width, out_channels]
        weight = self._get_linearized_weight()
        # weight.shape == [out_channels, kernel_width * in_channels]

        batch_size = X.size(1)
        if kernel_width > 1:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = X.new(kernel_width, batch_size, X.shape[2])
                input_buffer = input_buffer.zero_()

            # Shift buffer to remove the oldest step. The input buffer
            # records the last `kernel_size` steps of X.
            # Last step in the buffer is the current latest step
            input_buffer = torch.cat([input_buffer[1:], X[-1:]], dim=0)

            self._set_input_buffer(incremental_state, input_buffer)
            X = input_buffer

        X = X.transpose(0, 1).contiguous()
        # X.shape == [batch_size, kernel_width, in_channels]

        X = X.view(batch_size, -1)
        # X.shape == [batch_size, kernel_width * in_channels]

        output = F.linear(X, weight, self.bias)
        # output.shape == [batch_size, out_channels]

        output = output.view(batch_size, 1, -1)
        # output.shape == [batch_size, 1, out_channels]

        output = output.transpose(0, 1)
        # output.shape == [1, batch_size, out_channels]

        return output

    def _forward_multiple_steps(self, X, incremental_state):
        # X.shape == [seq_len, batch_size, in_channels]
        kernel_width = self.kernel_size[0]

        # Retrieve the previous `kernel_width` steps of X
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            X = torch.cat([input_buffer, X], dim=0)

        # Now that we have the required history, we simply put X through
        # the standard ConvTBC.
        output = super().forward(X)
        # output.shape == [buffer_len + seq_len + padding, batch_size, out_channels]

        # Remove future timesteps added by padding. We usually pad the input X
        # with `padding == kernel_size - 1` zeros so that the output retains
        # the same time dimension, once the right pads are removed
        if self.kernel_size[0] > 1 and self.padding[0] > 0:
            output = output[:-self.padding[0], :, :]
        # output.shape == [buffer_len + seq_len, batch_size, out_channels]

        # We also need to remove the extra buffer on the left
        if input_buffer is not None:
            buffer_len = input_buffer.shape[0]
            output = output[buffer_len:, :, :]
        # output.shape == [seq_len, batch_size, out_channels]

        # We store the previous `kernel_width` steps of X
        input_buffer = X[-kernel_width:]
        self._set_input_buffer(incremental_state, input_buffer)

        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            # The kernel width, e.g. 4
            kw = self.kernel_size[0]

            # self.weight.shape == [kernel_width, in_channels, out_channels]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None
