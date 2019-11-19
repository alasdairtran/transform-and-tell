# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tell.utils import get_incremental_state, set_incremental_state

from .unfold import unfold1d


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class DynamicConv1dTBC(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=False, conv_bias=False,
                 query_size=None, in_proj=False):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding

        if in_proj:
            self.weight_linear = Linear(
                self.input_size, self.input_size + num_heads * kernel_size * 1)
        else:
            self.weight_linear = Linear(
                self.query_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, X, incremental_state=None, query=None, unfold=None):
        '''Assuming the input, X, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            X: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''
        # X.shape == [seq_len, batch_size, input_dim]

        if incremental_state is not None:
            prev_X = self._get_input_buffer(incremental_state)
            if prev_X is not None:
                X = torch.cat([prev_X, X], dim=0)
            self._set_input_buffer(incremental_state, X[-self.kernel_size+1:])

        # use unfold mode as default for long sequence to save memory
        unfold = X.size(0) > 512 if unfold is None else unfold
        # unfold = unfold or (incremental_state is not None)
        assert query is None or not self.in_proj

        if query is None:
            query = X

        if unfold:
            output = self._forward_unfolded(X, incremental_state, query)
        else:
            output = self._forward_expanded(X, incremental_state, query)

        # Remove the history:
        if incremental_state is not None and prev_X is not None:
            output = output[prev_X.shape[0]:]

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        return output

    def _forward_unfolded(self, X, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        # X.shape == query.shape == [seq_len, batch_size, input_dim]

        T, B, C = X.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        # It seems to get the kernel weight, we can either project X or query.
        # We project the query by default.
        if self.in_proj:
            proj = self.weight_linear(X)
            X = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H *
                                 K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)
            # weight.shape == [seq_len * batch_size * n_heads, kernel_size]
            # Every word in a sequence will have a unique kernel of size
            # [n_heads, kernel_size]

        # renorm_padding is only implemented in _forward_expanded
        assert not self.renorm_padding

        padding_l = self.padding_l
        if K > T and padding_l == K-1:
            weight = weight.narrow(1, K-T, T)
            K, padding_l = T, T-1
        # For even padding, left padding should be K/2. Right padding will
        # automatically be K - padding_l - 1. To prevent our input from
        # seeing the future, padding needs to be K - 1 (and there will be
        # no right padding). This is what we always want in a decoder.

        # X.shape == query.shape == [seq_len, batch_size, input_dim]
        X_unfold = unfold1d(X, K, padding_l, 0)
        # X.shape == query.shape == [seq_len, batch_size, input_dim, kernel_size]

        X_unfold = X_unfold.view(T*B*H, R, K)
        # X.shape == query.shape == [seq_len * batch_size * n_heads, head_size, kernel_size]

        # We can think of the weight as the important of each kernel item.
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
            # weight.shape == [seq_len * batch_size * n_heads, kernel_size]

        # This doesn't do anything for self.in_proj == False. The last
        # dimension should already have length `kernel_size`.
        weight = weight.narrow(1, 0, K)
        # weight.shape == [seq_len * batch_size * n_heads, kernel_size]

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout,
                           training=self.training, inplace=False)

        weight = weight.unsqueeze(2)
        # X_unfold.shape == [seq_len * batch_size * n_heads, head_size, c_kernel_size]
        # weight.shape == [seq_len * batch_size * n_heads, c_kernel_size, 1]

        output = torch.bmm(X_unfold, weight)
        # output.shape == [seq_len * batch_size * n_heads, head_size, 1]

        output = output.view(T, B, C)
        # output.shape == [seq_len, batch_size, n_heads * head_size]

        return output

    def _old_forward_unfolded(self, X, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        # X.shape == query.shape == [seq_len, batch_size, input_dim]

        T, B, C = X.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        # It seems to get the kernel weight, we can either project X or query.
        # We project the query by default.
        if self.in_proj:
            proj = self.weight_linear(X)
            X = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H *
                                 K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)
            # weight.shape == [seq_len * batch_size * n_heads, kernel_size]
            # Every word in a sequence will have a unique kernel of size
            # [n_heads, kernel_size]

        # renorm_padding is only implemented in _forward_expanded
        assert not self.renorm_padding or incremental_state is not None

        if incremental_state is not None:
            # The input buffer is simply X
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = X.new()
            X_unfold = torch.cat([input_buffer, X.unsqueeze(3)], dim=3)
            # X_unfold.shape == [seq_len, batch_size, input_dim, c_kernel_size]

            if self.kernel_size > 1:
                self._set_input_buffer(
                    incremental_state, X_unfold[:, :, :, -self.kernel_size+1:])
            X_unfold = X_unfold.view(T*B*H, R, -1)
            # X_unfold.shape == [seq_len * batch_size * n_heads, head_size, c_kernel_size]
            # Each head is responsible for a patch in the input dim.

        else:
            padding_l = self.padding_l
            if K > T and padding_l == K-1:
                weight = weight.narrow(1, K-T, T)
                K, padding_l = T, T-1
            # For even padding, left padding should be K/2. Right padding will
            # automatically be K - padding_l - 1. To prevent our input from
            # seeing the future, padding needs to be K - 1 (and there will be
            # no right padding). This is what we always want in a decoder.

            # X.shape == query.shape == [seq_len, batch_size, input_dim]
            X_unfold = unfold1d(X, K, padding_l, 0)
            # X.shape == query.shape == [seq_len, batch_size, input_dim, kernel_size]

            X_unfold = X_unfold.view(T*B*H, R, K)
            # X.shape == query.shape == [seq_len * batch_size * n_heads, head_size, kernel_size]

        # We can think of the weight as the important of each kernel item.
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
            # weight.shape == [seq_len * batch_size * n_heads, kernel_size]

        # This doesn't do anything for self.in_proj == False. The last
        # dimension should already have length `kernel_size`.
        weight = weight.narrow(1, 0, K)
        # weight.shape == [seq_len * batch_size * n_heads, kernel_size]

        # Once we have seen `kernel_size` words, this operation has no effect.
        # K is now the actual kernel size.
        if incremental_state is not None:
            weight = weight[:, -X_unfold.size(2):]
            K = weight.size(1)
            # weight.shape == [seq_len * batch_size * n_heads, c_kernel_size]

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout,
                           training=self.training, inplace=False)

        weight = weight.unsqueeze(2)
        # X_unfold.shape == [seq_len * batch_size * n_heads, head_size, c_kernel_size]
        # weight.shape == [seq_len * batch_size * n_heads, c_kernel_size, 1]

        output = torch.bmm(X_unfold, weight)
        # output.shape == [seq_len * batch_size * n_heads, head_size, 1]

        output = output.view(T, B, C)
        # output.shape == [seq_len, batch_size, n_heads * head_size]

        return output

    def _forward_expanded(self, X, incremental_state, query):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = X.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(X)
            X = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H *
                                 K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout,
                               training=self.training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        X = X.view(T, B*H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
            weight_expanded.as_strided(
                (B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(
                weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when the kernel is larger than the length
            if K > T and P == K-1:
                weight = weight.narrow(2, K-T, T)
                K, P = T, T-1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(
                B*H, T, T+K-1, requires_grad=False)
            weight_expanded.as_strided(
                (B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T

        output = torch.bmm(weight_expanded, X)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def extra_repr(self):
        s = '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, conv_bias={}, renorm_padding={}, in_proj={}'.format(
            self.input_size, self.kernel_size, self.padding_l,
            self.num_heads, self.weight_softmax, self.conv_bias is not None, self.renorm_padding,
            self.in_proj,
        )

        if self.query_size != self.input_size:
            s += ', query_size={}'.format(self.query_size)
        if self.weight_dropout > 0.:
            s += ', weight_dropout={}'.format(self.weight_dropout)
        return s
