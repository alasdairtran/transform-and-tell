import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from newser.modules.linear import GehringLinear

from .scalar_bias import scalar_bias


class SingleHeadAttention(nn.Module):
    """Single-head attention that supports Gating and Downsampling."""

    def __init__(self, out_channels, embed_dim, head_dim, head_index, dropout=0.,
                 bias=True, project_input=True, gated=False, downsample=False,
                 num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.head_index = head_index
        self.head_dim = head_dim
        self.project_input = project_input
        self.gated = gated
        self.downsample = downsample
        self.num_heads = num_heads
        self.projection = None

        k_layers = []
        v_layers = []
        if self.downsample:
            k_layers.append(Downsample(self.head_index))
            v_layers.append(Downsample(self.head_index))
            out_proj_size = self.head_dim
        else:
            out_proj_size = self.head_dim * self.num_heads
        if self.gated:
            k_layers.append(GatedLinear(
                self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = GatedLinear(
                self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(GatedLinear(
                self.embed_dim, out_proj_size, bias=bias))
        else:
            k_layers.append(GehringLinear(
                self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = GehringLinear(
                self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(GehringLinear(
                self.embed_dim, out_proj_size, bias=bias))

        self.in_proj_k = nn.Sequential(*k_layers)
        self.in_proj_v = nn.Sequential(*v_layers)

        if self.downsample:
            self.out_proj = GehringLinear(
                out_proj_size, self.head_dim, bias=bias)
        else:
            self.out_proj = GehringLinear(
                out_proj_size, out_channels, bias=bias)

        self.scaling = self.head_dim**-0.5

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, use_scalar_bias=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        orginal_src_len, batch_size, out_channels = key.shape
        tgt_len = query.shape[0]
        assert list(query.shape) == [tgt_len, batch_size, out_channels]
        assert key.shape == value.shape

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == batch_size
            assert key_padding_mask.shape[1] == orginal_src_len

        if self.downsample:
            size = batch_size
        else:
            size = batch_size * self.num_heads

        k = key
        v = value
        q = query
        src_len = orginal_src_len
        if self.project_input:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)
            src_len = k.shape[0]
        q = q * self.scaling

        if not self.downsample:
            q = q.view(tgt_len, size, self.head_dim)
            k = k.view(src_len, size, self.head_dim)
            v = v.view(src_len, size, self.head_dim)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # v.shape = [batch_size, src_len, embed_dim]

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights.shape == [batch_size, tgt_len, src_len]

        # Note that we should only mask future timesteps when we're doing
        # self-attention.
        if mask_future_timesteps:
            # If we're not in incremental mode, the attention weight is a
            # square matrix, and the diagonal is the attention on the current
            # step, while the upper triangle is the attention on future steps.
            # Furthermore, when using downsampling, the attention weight will
            # have fewer columns than src_len.
            if query.shape == key.shape:
                attn_weights = self._mask_future_full(attn_weights)

            # Otherwise, we assume that we're in incremental mode and we only
            # have a partial query. The attention weight matrix now has more
            # columns than rows.
            else:
                attn_weights = self._mask_future_partial(
                    attn_weights, orginal_src_len)

        # Give our model the option to attend to not attend to anything at all
        # (i.e. a zero placeholder is added at the beginning of the source).
        src_size = src_len
        if use_scalar_bias:
            attn_weights = scalar_bias(attn_weights, 2)
            # attn_weights.shape == [batch_size, tgt_len, 1 + src_len]
            v = scalar_bias(v, 1)
            # v.shape = [batch_size, 1 + src_len, embed_dim]
            src_size += 1

        if key_padding_mask is not None:
            # don't attend to padding symbols
            if key_padding_mask.max() > 0:
                if self.downsample:
                    attn_weights = attn_weights.view(
                        batch_size, 1, tgt_len, src_len)
                else:
                    attn_weights = attn_weights.view(
                        size, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -math.inf,
                )
                attn_weights = attn_weights.view(size, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        # attn.shape == [batch_size, tgt_len, embed_dim]

        if self.downsample:
            attn = attn.transpose(0, 1).contiguous().view(
                tgt_len, batch_size, self.head_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(
                tgt_len, batch_size, self.embed_dim)

        attn = self.out_proj(attn)
        # attn.shape == [tgt_len, batch_size, out_channels]
        # attn_weights.shape == [batch_size, tgt_len, 1 + src_len]

        return attn, attn_weights

    def _mask_future_full(self, attn_weights):
        tgt_len = attn_weights.shape[1]

        # Zero out the upper triangle, including the diagonal (we don't
        # attend to ourself, but only to past words)
        ones = attn_weights.data.new([1]).expand(tgt_len, tgt_len).clone()
        mask = torch.tril(ones, diagonal=-1)
        mask = mask[:, ::self.head_index + 1 if self.downsample else 1]
        mask = mask.unsqueeze(0)
        attn_weights *= mask

        # Give all of the zero-out entries a value of -infinity. This means
        # we'll get a probability of zero after applying softmax.
        offset = attn_weights.data.new([-math.inf])
        offset = offset.expand(tgt_len, tgt_len).clone()
        offset = torch.triu(offset, diagonal=0)
        offset = offset[:, ::self.head_index + 1 if self.downsample else 1]
        offset = offset.unsqueeze(0)
        attn_weights += offset

        return attn_weights

    def _mask_future_partial(self, attn_weights, orginal_src_len):
        """Basically the same as _mask_future_full, but we can deal with
        non-square attention matrices."""
        _, tgt_len, _ = attn_weights.shape

        # For the last row, we want to zero out the last entry (we don't
        # attend to ourself). For the second-to-last row, we want to zero out
        # the last two entries. And so on. The diagonal calculation will
        # help us construct this mask.
        ones = attn_weights.data.new([1])
        ones = ones.expand(tgt_len, orginal_src_len).clone()
        diagonal = orginal_src_len - tgt_len - 1
        mask = torch.tril(ones, diagonal=diagonal)
        mask = mask[:, ::self.head_index + 1 if self.downsample else 1]
        mask = mask.unsqueeze(0)
        attn_weights *= mask

        # Give all of the zero-out entries a value of -infinity. This means
        # we'll get a probability of zero after applying softmax.
        offset = attn_weights.data.new([-math.inf])
        offset = offset.expand(tgt_len, orginal_src_len).clone()
        diagonal = orginal_src_len - tgt_len
        offset = torch.triu(offset, diagonal=diagonal)
        offset = offset[:, ::self.head_index + 1 if self.downsample else 1]
        offset = offset.unsqueeze(0)
        attn_weights += offset

        return attn_weights


class Downsample(nn.Module):
    """Selects every nth element, where n is the index. """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[::self.index+1]


def GatedLinear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(
        GehringLinear(in_features, out_features*4, dropout, bias),
        nn.GLU(),
        GehringLinear(out_features*2, out_features*2, dropout, bias),
        nn.GLU(),
        GehringLinear(out_features, out_features, dropout, bias)
    )
