import math

import torch.nn.functional as F

from tell.utils import strip_pad

from .base import Criterion


@Criterion.register('adaptive_loss')
class AdaptiveLoss(Criterion):
    """Create the loss for the adaptive softmax approximation.

    This is an implementation of the loss function accompanying the adaptive
    softmax approximation for graphical processing units (GPU), described in
    the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, padding_idx=1):
        super().__init__()
        self.padding_idx = padding_idx
        # normalize gradients by the number of sentences in a batch
        # (default is to normalize by number of tokens)
        self.sentence_avg = False

    def forward(self, adaptive_softmax, net_output, decoder_target, reduction='sum'):
        """Compute the loss for the given sample.

        Reduction can be 'sum' or None

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        orig_target = decoder_target
        # orig_target.shape == [batch_size, seq_len]

        orig_target = orig_target.reshape(-1)
        # orig_target.shape == [batch_size * seq_len]

        batch_size = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)
        # len(target) == len(logits) == n_clusters
        # logits[i].shape == [batch_size * seq_len, cluster_size]
        # target[i].shape == [batch_size * seq_len]

        loss = net_output[0].new(
            1 if reduction == 'sum' else batch_size).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0 and target[i].max()
                        <= logits[i].size(1))
                loss += F.cross_entropy(logits[i], target[i], ignore_index=self.padding_idx,
                                        reduction=reduction)

        orig = strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = decoder_target.size(
            0) if self.sentence_avg else ntokens
        logging_output = {
            'loss': loss.data.item() if reduction == 'sum' else loss.data,
            'ntokens': ntokens,
            'nsentences': batch_size,
            'sample_size': sample_size,
        }
        loss = loss
        return loss, sample_size

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'nll_loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
