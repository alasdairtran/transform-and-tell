import torch
import torch.nn as nn

from tell.modules.linear import GehringLinear

from .downsampled_single_head import SingleHeadAttention


class DownsampledMultiHeadAttention(nn.ModuleList):
    """Multi-headed attention with Gating and Downsampling."""

    def __init__(self, out_channels, embed_dim, num_heads, dropout=0., bias=True,
                 project_input=True, gated=False, downsample=False):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.gated = gated
        self.project_input = project_input
        assert self.head_dim * num_heads == embed_dim

        if self.downsample:
            attention_heads = []
            for index in range(self.num_heads):
                attention_heads.append(
                    SingleHeadAttention(
                        out_channels, self.embed_dim, self.head_dim, index,
                        self.dropout, bias, self.project_input, self.gated,
                        self.downsample, self.num_heads,
                    )
                )
            super().__init__(modules=attention_heads)
            self.out_proj = GehringLinear(embed_dim, out_channels, bias=bias)
        else:
            # either we have a list of attention heads, or just one attention head
            # if not being downsampled, we can do the heads with one linear layer instead of separate ones
            super().__init__()
            self.attention_module = SingleHeadAttention(
                out_channels, self.embed_dim, self.head_dim, 1, self.dropout,
                bias, self.project_input, self.gated, self.downsample, self.num_heads,
            )

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, use_scalar_bias=False):
        src_len, batch_size, embed_dim = key.shape
        tgt_len = query.shape[0]
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, batch_size, embed_dim]
        assert key.shape == value.shape

        src_size = src_len
        if use_scalar_bias:
            src_size += 1

        attn = []
        attn_weights = []
        if self.downsample:
            for attention_head_number in range(self.num_heads):
                # call the forward of each attention head
                _attn, _attn_weight = self[attention_head_number](
                    query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias,
                )
                attn.append(_attn)
                attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn = self.out_proj(full_attn)
            return full_attn, attn_weights[0].clone()
        else:
            _attn, _attn_weight = self.attention_module(
                query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias,
            )
            # _attn.shape == [tgt_len, batch_size, out_channels]
            # _attn_weight.shape == [batch_size, tgt_len, 1 + src_len]
            attn.append(_attn)
            attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn_weights = torch.cat(attn_weights)
            full_attn_weights = full_attn_weights.view(
                batch_size, self.num_heads, tgt_len, src_size)
            full_attn_weights = full_attn_weights.sum(dim=1) / self.num_heads

            # full_attn.shape == [tgt_len, batch_size, out_channels]
            # full_attn_weights.shape == [batch_size, tgt_len, src_size]
            return full_attn, full_attn_weights
