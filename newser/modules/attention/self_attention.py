import torch
import torch.nn as nn

from newser.modules.linear import GehringLinear
from newser.utils import get_incremental_state, set_incremental_state

from .downsampled_multi_head import DownsampledMultiHeadAttention


class SelfAttention(nn.Module):
    """Construct self-attention module.

    Self-attention is used in the decoder as it generates text. At this stage,
    there are two key restrictions. First, we do not use dropout on the
    attention weights, since we want to make use of all the words generated so
    far. Secondly, we ensure that the future timesteps are masked so that we
    can't peek into the future.

    Parameters
    ----------
    out_channels : int

    """

    def __init__(self, out_channels, embed_dim, num_heads, project_input=False,
                 gated=False, downsample=False, weight_norm=True):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(
            out_channels, embed_dim, num_heads, dropout=0, bias=True,
            project_input=project_input, gated=gated, downsample=downsample)
        self.in_proj_q = GehringLinear(
            out_channels, embed_dim, weight_norm=weight_norm)
        self.in_proj_k = GehringLinear(
            out_channels, embed_dim, weight_norm=weight_norm)
        self.in_proj_v = GehringLinear(
            out_channels, embed_dim, weight_norm=weight_norm)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, X, incremental_state=None):
        residual = X
        # residual.shape == X.shape == [seq_len, batch_size, out_channels]
        query = self.in_proj_q(X)
        key = self.in_proj_k(X)
        value = self.in_proj_v(X)

        # When the incremental state is enabled, we assume that all the
        # previous key and value (the entire history) will be used in the
        # next step.
        if incremental_state is not None:
            prev_key, prev_value = self._get_history(incremental_state)
            if prev_key is not None:
                key = torch.cat([prev_key, key], dim=0)
            if prev_value is not None:
                value = torch.cat([prev_value, value], dim=0)
            self._save_history(incremental_state, key, value)
        # key.shape == value.shape == [prev_seq_len, batch_size, embed_dim]

        # No need to mask future timestep if query sequence contains only 1 step
        # mask_future_timesteps = query.shape[0] > 1
        X, _ = self.attention(query, key, value,
                              mask_future_timesteps=True,
                              use_scalar_bias=True)
        # X.shape == [seq_len, batch_size, out_channels]
        return self.ln(X + residual)

    def _save_history(self, incremental_state, key, value):
        set_incremental_state(self, incremental_state, 'key', key)
        set_incremental_state(self, incremental_state, 'value', value)

    def _get_history(self, incremental_state):
        key = get_incremental_state(self, incremental_state, 'key')
        value = get_incremental_state(self, incremental_state, 'value')
        return key, value
