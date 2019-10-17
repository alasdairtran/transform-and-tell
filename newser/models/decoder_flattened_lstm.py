# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from newser.modules import AdaptiveSoftmax, GehringLinear
from newser.modules.token_embedders import AdaptiveEmbedding
from newser.utils import eval_str_list

from .decoder_flattened import Decoder


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = GehringLinear(
            input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = GehringLinear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        # attn_scores.shape == [src_len, bsz]

        encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
        # encoder_padding_mask.shape == [src_len, bsz]

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


@Decoder.register('lstm_decoder_flattened')
class LSTMDecoder(Decoder):
    """
    DynamicConv decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`DynamicConvDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, vocab, embedder: TextFieldEmbedder, num_layers,
                 hidden_size, dropout, share_decoder_input_output_embed,
                 vocab_size=None, adaptive_softmax_cutoff=None,
                 tie_adaptive_weights=False, adaptive_softmax_dropout=0,
                 tie_adaptive_proj=False, adaptive_softmax_factor=0,
                 article_embed_size=1024, image_embed_size=2048,
                 namespace='target_tokens'):
        super().__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        vocab_size = vocab_size or vocab.get_vocab_size(namespace)
        self.dropout = dropout
        self.share_input_output_embed = share_decoder_input_output_embed

        input_embed_dim = embedder.get_output_dim()
        embed_dim = input_embed_dim
        output_embed_dim = input_embed_dim

        self.layers = nn.ModuleList([])
        self.h = nn.ParameterList([])
        self.c = nn.ParameterList([])
        for layer in range(num_layers):
            input_size = hidden_size + embed_dim if layer == 0 else hidden_size
            rnn = LSTMCell(input_size=input_size, hidden_size=hidden_size)
            self.layers.append(rnn)
            self.h.append(nn.Parameter(torch.zeros(1, hidden_size)))
            self.c.append(nn.Parameter(torch.zeros(1, hidden_size)))

        self.image_attention = AttentionLayer(
            hidden_size, image_embed_size, hidden_size, bias=True)

        self.article_attention = AttentionLayer(
            hidden_size, article_embed_size, hidden_size, bias=True)

        self.attn_proj = GehringLinear(hidden_size * 2, hidden_size)

        self.embedder = embedder

        self.adaptive_softmax = None

        self.project_out_dim = GehringLinear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not tie_adaptive_weights else None

        if adaptive_softmax_cutoff is not None:
            adaptive_inputs = None
            if isinstance(embedder, AdaptiveEmbedding):
                adaptive_inputs = embedder
            elif hasattr(embedder, 'token_embedder_adaptive'):
                adaptive_inputs = embedder.token_embedder_adaptive
            elif tie_adaptive_weights:
                raise ValueError('Cannot locate adaptive_inputs.')
            self.adaptive_softmax = AdaptiveSoftmax(
                vocab_size,
                output_embed_dim,
                eval_str_list(adaptive_softmax_cutoff, type=int),
                dropout=adaptive_softmax_dropout,
                adaptive_inputs=adaptive_inputs,
                factor=adaptive_softmax_factor,
                tie_proj=tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(vocab_size, output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0,
                            std=output_embed_dim ** -0.5)

    def forward(self, prev_target, contexts, incremental_state=None,
                use_layers=None, **kwargs):
        """
        Args:
            prev_target (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        # embed tokens and positions
        X = self.embedder(prev_target, incremental_state=incremental_state)
        X = F.dropout(X, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        X = X.transpose(0, 1)

        T, B, _ = X.shape
        C = contexts['image'].shape[0]
        S = contexts['article'].shape[0]
        n_layers = len(self.layers)

        prev_hiddens = [self.h[i].expand(B, -1) for i in range(n_layers)]
        prev_cells = [self.c[i].expand(B, -1) for i in range(n_layers)]
        input_feed = X.new_zeros(B, self.hidden_size)
        image_attn_scores = X.new_zeros(C, T, B)
        article_attn_scores = X.new_zeros(S, T, B)
        outs = []

        for step in range(T):
            # input feeding: concatenate context vector from previous time step
            rnn_input = torch.cat((X[step, :, :], input_feed), dim=1)
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(rnn_input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                rnn_input = F.dropout(hidden, p=self.dropout,
                                      training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            image_out, image_attn_scores[:, step, :] = self.image_attention(
                hidden, contexts['image'], contexts['image_mask'])

            article_out, article_attn_scores[:, step, :] = self.article_attention(
                hidden, contexts['article'], contexts['article_mask'])

            out = torch.cat([image_out, article_out], dim=1)
            out = F.dropout(out, p=self.dropout, training=self.training)
            # out.shape == [B, hidden_size * 2]

            out = self.attn_proj(out)
            # out.shape == [B, hidden_size]

            input_feed = out

            outs.append(out)

        # collect outputs across time steps
        X = torch.cat(outs, dim=0).view(T, B, self.hidden_size)

        # T x B x C -> B x T x C
        X = X.transpose(1, 0)

        if self.project_out_dim is not None:
            X = self.project_out_dim(X)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                X = F.linear(
                    X, self.embedder.token_embedder_bert.word_embeddings.weight)
            else:
                X = F.linear(X, self.embed_out)

        return X, {'attn': None, 'inner_states': None}

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            target = sample['target'] if sample else None
            out = self.adaptive_softmax.get_log_prob(
                net_output[0], target)
            return out.exp() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
