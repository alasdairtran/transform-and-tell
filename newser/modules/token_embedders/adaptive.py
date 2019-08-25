import math
from typing import List

import torch
import torch.nn as nn
import torch.onnx.operators
from allennlp.modules.token_embedders import TokenEmbedder
from overrides import overrides


@TokenEmbedder.register('adaptive')
class AdaptiveEmbedding(TokenEmbedder):
    """Adaptive input representation, proposed by Baevski & Auli (2019).

    Adaptive input representations for neural language modeling extend the
    adaptive softmax of Grave et al. (2017) to input representations of
    variable capacity. See https://openreview.net/forum?id=ByxZX20qFQ.
    """

    def __init__(self, vocab, namespace, padding_idx: int,
                 initial_dim: int, factor: float, output_dim: int,
                 cutoff: List[int], vocab_size: int = None, scale_embeds=False):
        super().__init__()

        vocab_size = vocab_size or vocab.get_vocab_size(namespace)
        if not cutoff or vocab_size > cutoff[-1]:
            cutoff.append(vocab_size)

        assert vocab_size == cutoff[-1], \
            f'Cutoff {cutoff[-1]} is larger than vocab size {vocab_size}.'

        self.cutoff = cutoff
        self.embed_size = output_dim
        self.padding_idx = padding_idx
        self.embeddings = nn.ModuleList()
        self.embed_scale = math.sqrt(output_dim) if scale_embeds else 1

        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            vocab_size = self.cutoff[i] - prev
            embed_size = int(initial_dim // (factor ** i))
            embed = nn.Embedding(vocab_size, embed_size, padding_idx)
            projection = nn.Linear(embed_size, output_dim, bias=False)
            seq = nn.Sequential(embed, projection)
            self.embeddings.append(seq)

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                std = math.sqrt(1 / m.weight.shape[1])
                m.weight.data.normal_(mean=0, std=std)
                m.weight.data[padding_idx].fill_(0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        # Recursively initialize weights of all children
        self.apply(init_weights)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, X: torch.Tensor, incremental_state=None):
        result_shape = X.shape + (self.embed_size,)
        result = self._float_tensor.new_zeros(result_shape)

        for i in range(len(self.cutoff)):
            mask = X < self.cutoff[i]
            if i > 0:
                mask.mul_(X >= self.cutoff[i - 1])
                chunk_input = X[mask] - self.cutoff[i - 1]
            else:
                chunk_input = X[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)

        result = self.embed_scale * result
        return result

    @overrides
    def get_output_dim(self) -> int:
        return self.embed_size
