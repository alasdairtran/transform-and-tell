import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import TiedLinear


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, n_classes):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = TiedLinear(tied_emb, transpose=False)
        if input_dim != emb_dim:
            linear = nn.Linear(input_dim, emb_dim, bias=False)
            self.word_proj = nn.Sequential(linear, self.word_proj)

        self.n_classes = n_classes
        if n_classes > 0:
            self.class_proj = nn.Linear(input_dim, n_classes, bias=False)
        self.out_dim = self.num_words + n_classes

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, X):
        # Flatten out the batch and time dimensions
        input_size = functools.reduce(operator.mul, X.shape[:-1], 1)
        X = X.view(input_size, -1)

        out = [self.word_proj(X)]
        if self.n_classes:
            out.append(self.class_proj(X))

        out = torch.cat(out, dim=1)
        return out


class AdaptiveSoftmax(nn.Module):
    """Efficient softmax approximation for GPU proposed by Grave et al. (2016).

    Since rare words cannot be learned very well, we share the state of the
    hidden layer across clusters and simply reduce the input size of the
    classifiers by applying a projection matrix. Typically, the projection
    matrix for the tail cluster reduces the size from d to d/4. See the
    original proposal in "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309).

    Parameters
    ----------
    cutoff : List[int]
        For example, the cutoff used in Dauphin et al. (2017) for the
        WikiText-103 is [10000, 20000, 200000]

    adaptive_inputs : AdaptiveEmbedding
        If given, use the adaptive input representations as proposed by
        Baevski & Auli (2019).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=4.,
                 adaptive_inputs=None, tie_proj=False):
        super().__init__()

        if not cutoff or vocab_size > cutoff[-1]:
            cutoff.append(vocab_size)

        assert vocab_size == cutoff[-1], \
            f'Cutoff {cutoff[-1]} is larger than vocab size {vocab_size}.'

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor
        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            embed_weight = adaptive_inputs.weights_for_band(0)
            n_tails = len(cutoff) - 1
            self.head = TiedHeadModule(embed_weight, input_dim, n_tails)
        else:
            self.head = nn.Linear(input_dim, output_dim, bias=False)

        self._make_tail(True, adaptive_inputs, tie_proj)

        def init_weights(m):
            if hasattr(m, 'weight') and not \
                    isinstance(m, TiedLinear) and not \
                    isinstance(m, TiedHeadModule):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('version', torch.LongTensor([1]))
        # versions prior to 1 had a bug that offset indices on the head by 1
        self.buggy_offset = 0

    def _make_tail(self, fix_exponent, adaptive_inputs=None, tie_proj=False):
        extra_denom = 1 if fix_exponent else 0

        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + extra_denom))

            tied_emb, tied_proj = adaptive_inputs.weights_for_band(i + 1) \
                if adaptive_inputs is not None else (None, None)

            # There was a bug in the original Fairseq implementation, where
            # it assumes that self.input_dim has the same dimension as
            # tied_proj.shape[0] (the output dimension of the adaptive embeddings).
            # The change below removes this assumption.
            if tied_proj is not None:
                if tie_proj:
                    proj = TiedLinear(tied_proj, transpose=True)
                else:
                    proj = nn.Linear(
                        self.input_dim, tied_proj.shape[1], bias=False)
            else:
                proj = nn.Linear(self.input_dim, dim, bias=False)

            m = nn.Sequential(
                proj,
                nn.Dropout(self.dropout),
                nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False,
                ) if tied_emb is None else TiedLinear(tied_emb, transpose=False),
            )

            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            self.buggy_offset = 1
            self._make_tail(False)
            state_dict[version_name] = torch.LongTensor([1])

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i - self.buggy_offset

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, X, target):
        """
        Args:
            X: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        X = X.contiguous().view(-1, X.size(-1))
        X = F.dropout(X, p=self.dropout, training=self.training)

        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(X)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](
                    X.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, X, target=None):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        # We don't support the target argument for now.
        assert target is None

        batch_size, seq_len, dim = X.size()
        X = X.contiguous().view(-1, dim)

        head_y = self.head(X)
        # log_probs = head_y.new_zeros(X.size(0), self.vocab_size)

        head_size = self.cutoff[0] + len(self.tail)
        head_log_probs = self.lsm(head_y)
        log_probs_list = [head_log_probs[:, :self.cutoff[0]]]

        if len(self.tail) > 0:
            tail_priors = head_log_probs[:, self.cutoff[0]:head_size]

            for i in range(len(self.tail)):
                tail_i = self.lsm(self.tail[i](X))
                tail_i = tail_i + tail_priors[:, i, None]
                log_probs_list.append(tail_i)

        log_probs = torch.cat(log_probs_list, dim=1)
        log_probs = log_probs.view(batch_size, seq_len, self.vocab_size)
        return log_probs
