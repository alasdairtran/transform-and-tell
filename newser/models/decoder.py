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
from newser.utils import eval_str_list, fill_with_neg_inf


class Decoder(Registrable, nn.Module):
    pass


class DecoderLayer(Registrable, nn.Module):
    pass


@Decoder.register('dynamic_conv_decoder')
class DynamicConvDecoder(Decoder):
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
                 decoder_kernel_size_list, adaptive_softmax_cutoff=None,
                 tie_adaptive_weights=False, adaptive_softmax_dropout=0,
                 tie_adaptive_proj=False, adaptive_softmax_factor=0, decoder_layers=6,
                 context_embed_sizes=[], final_norm=True, padding_idx=0, namespace='target_tokens',
                 vocab_size=None):
        super().__init__()
        self.vocab = vocab
        vocab_size = vocab_size or vocab.get_vocab_size(namespace)
        self.dropout = dropout
        self.share_input_output_embed = share_decoder_input_output_embed

        input_embed_dim = embedder.get_output_dim()
        embed_dim = input_embed_dim
        output_embed_dim = input_embed_dim

        padding_idx = padding_idx
        self.max_target_positions = max_target_positions

        self.embedder = embedder

        self.project_in_dim = GehringLinear(
            input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DynamicConvDecoderLayer(embed_dim, decoder_conv_dim, decoder_glu,
                                    decoder_conv_type, weight_softmax, decoder_attention_heads,
                                    weight_dropout, dropout, relu_dropout, input_dropout,
                                    decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                                    context_embed_sizes, kernel_size=decoder_kernel_size_list[i])
            for i in range(decoder_layers)
        ])

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

    def forward(self, prev_target, contexts, context_masks, incremental_state=None,
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
        attn = None

        inner_states = [X]

        # decoder layers
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(
                    X,
                    contexts,
                    context_masks,
                    incremental_state,
                )
                inner_states.append(X)

        if self.normalize:
            X = self.layer_norm(X)

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

        return X, {'attn': attn, 'inner_states': inner_states}

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

    def filter_incremental_state(self, incremental_state, active_idx):
        if incremental_state is None:
            return
        for key in incremental_state:
            if 'DynamicConv1dTBC' in key:
                incremental_state[key] = incremental_state[key][:, active_idx]


@DecoderLayer.register('dynamic_conv')
class DynamicConvDecoderLayer(DecoderLayer):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        kernel_size: kernel size of the convolution
    """

    def __init__(self, decoder_embed_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 context_embed_sizes=[], kernel_size=0):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        if decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2*self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None
        if decoder_conv_type == 'lightweight':
            self.conv = LightweightConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                             weight_softmax=weight_softmax,
                                             num_heads=decoder_attention_heads,
                                             weight_dropout=weight_dropout)
        elif decoder_conv_type == 'dynamic':
            self.conv = DynamicConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=weight_softmax,
                                         num_heads=decoder_attention_heads,
                                         weight_dropout=weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = decoder_normalize_before

        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleList()
        self.context_attn_lns = nn.ModuleList()
        for embed_size in context_embed_sizes:
            self.context_attns.append(MultiHeadAttention(
                embed_size, decoder_attention_heads,
                dropout=attention_dropout,
            ))
            self.context_attn_lns.append(nn.LayerNorm(self.embed_dim))

        context_size = self.embed_dim * len(context_embed_sizes)
        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, X, contexts, context_masks, incremental_state):
        """
        Args:
            X (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act is not None:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        attn = None
        X_contexts = []
        for context, context_mask, context_attn, context_ln in zip(contexts, context_masks, self.context_attns, self.context_attn_lns):
            residual = X
            X_context = self.maybe_layer_norm(
                context_ln, X, before=True)
            X_context, attn = context_attn(
                query=X_context,
                key=context,
                value=context,
                key_padding_mask=context_mask,
                incremental_state=None,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            X_context = F.dropout(
                X_context, p=self.dropout, training=self.training)
            X_context = residual + X_context
            X_context = self.maybe_layer_norm(
                context_ln, X_context, after=True)
            X_contexts.append(X_context)

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)
        return X, attn

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        else:
            return X

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before)
