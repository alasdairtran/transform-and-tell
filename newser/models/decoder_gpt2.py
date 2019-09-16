# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.registrable import Registrable
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from newser.modules import (AdaptiveSoftmax, DynamicConv1dTBC, GehringLinear,
                            LightweightConv1dTBC, MultiHeadAttention)
from newser.modules.token_embedders import AdaptiveEmbedding
from newser.utils import eval_str_list, fill_with_neg_inf, softmax

from .decoder_flattened import Decoder, DecoderLayer


@Decoder.register('gpt2')
class GPT2Decoder(Decoder):
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

    def __init__(self, vocab, embedder: TextFieldEmbedder, max_target_positions, dropout,
                 share_decoder_input_output_embed,
                 decoder_output_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 adaptive_softmax_cutoff=None,
                 tie_adaptive_weights=False, adaptive_softmax_dropout=0,
                 tie_adaptive_proj=False, adaptive_softmax_factor=0, decoder_layers=6,
                 final_norm=True, padding_idx=0, namespace='target_tokens',
                 vocab_size=None, section_attn=False):
        super().__init__()
        self.vocab = vocab
        vocab_size = vocab_size or vocab.get_vocab_size(namespace)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.share_input_output_embed = share_decoder_input_output_embed

        input_embed_dim = embedder.get_output_dim()
        embed_dim = input_embed_dim
        self.embed_dim = embed_dim
        output_embed_dim = input_embed_dim

        padding_idx = padding_idx
        self.max_target_positions = max_target_positions

        self.embedder = embedder

        self.project_in_dim = GehringLinear(
            input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.image_attns, self.article_attns = nn.ModuleList(), nn.ModuleList()
        self.ln_1, self.ln_2 = nn.ModuleList(), nn.ModuleList()
        self.ln_3, self.ln_4 = nn.ModuleList(), nn.ModuleList()
        self.ln_5 = nn.ModuleList()
        self.fc1s, self.fc2s = nn.ModuleList(), nn.ModuleList()
        self.context_fcs = nn.ModuleList()
        C = 2048

        hidden_dim = embed_dim * 4
        for _ in range(decoder_layers):
            self.image_attns.append(MultiHeadAttention(
                embed_dim, decoder_attention_heads, kdim=C, vdim=C,
                dropout=attention_dropout))
            self.article_attns.append(MultiHeadAttention(
                embed_dim, decoder_attention_heads, kdim=1024, vdim=1024,
                dropout=attention_dropout))
            self.attentions.append(nn.MultiheadAttention(
                embed_dim, decoder_attention_heads, dropout=attention_dropout))
            self.feed_forwards.append(nn.Sequential(
                GehringLinear(embed_dim, hidden_dim),
                nn.ReLU(),
                GehringLinear(hidden_dim, embed_dim)))
            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_3.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_4.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_5.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.fc1s.append(GehringLinear(
                embed_dim, decoder_ffn_embed_dim))
            self.fc2s.append(GehringLinear(
                decoder_ffn_embed_dim, embed_dim))
            context_size = embed_dim * 2
            self.context_fcs.append(
                GehringLinear(context_size, embed_dim))

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
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)
        self.need_attn = True

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

        # if incremental_state is not None:
        #     X = X[:, -1:]

        if self.project_in_dim is not None:
            X = self.project_in_dim(X)

        X = F.dropout(X, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        X = X.transpose(0, 1)
        T, B, C = X.shape

        attn_mask = X.new_full((T, T), -float('inf'))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for ln_1, attn, ln_2, ff, image_attn, article_attn, ln_3, ln_4, fc_1, fc_2, ln_5, context_fc in zip(
                self.ln_1, self.attentions, self.ln_2, self.feed_forwards,
                self.image_attns, self.article_attns, self.ln_3, self.ln_4,
                self.fc1s, self.fc2s, self.ln_5, self.context_fcs):

            h = ln_1(X)
            x, _ = attn(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = x + h

            h = ln_2(h)
            x = ff(h)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = x + h

            X = h

            X_contexts = []

            # Image attention
            residual = X
            X_image = X
            X_image, attn = image_attn(
                query=X_image,
                key=contexts['image'],
                value=contexts['image'],
                key_padding_mask=contexts['image_mask'],
                incremental_state=None,
                static_kv=True,
                need_weights=(not self.training and self.need_attn))
            X_image = F.dropout(X_image, p=self.dropout,
                                training=self.training)
            X_image = residual + X_image
            X_image = ln_3(X_image)
            X_contexts.append(X_image)

            # Article attention
            residual = X
            X_article = X
            X_article, attn = article_attn(
                query=X_article,
                key=contexts['article'],
                value=contexts['article'],
                key_padding_mask=contexts['article_mask'],
                incremental_state=None,
                static_kv=True,
                need_weights=True)
            X_article = F.dropout(X_article, p=self.dropout,
                                  training=self.training)
            X_article = residual + X_article
            X_article = ln_4(X_article)
            X_contexts.append(X_article)

            X_context = torch.cat(X_contexts, dim=-1)
            X = context_fc(X_context)

            residual = X
            X = F.relu(fc_1(X))
            X = F.dropout(X, p=self.relu_dropout, training=self.training)
            X = fc_2(X)
            X = F.dropout(X, p=self.dropout, training=self.training)
            X = residual + X
            X = ln_5(X)

        # T x B x C -> B x T x C
        X = X.transpose(0, 1)

        if self.project_out_dim is not None:
            X = self.project_out_dim(X)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                X = F.linear(
                    X, self.embedder.token_embedder_bert.word_embeddings.weight)
            else:
                X = F.linear(X, self.embed_out)

        return X, {}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions
        # return min(self.max_target_positions, self.embedder.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # pylint: disable=access-member-before-definition
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(
                fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(
                self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

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
