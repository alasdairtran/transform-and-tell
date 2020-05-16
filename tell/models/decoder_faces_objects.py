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

from tell.modules import (AdaptiveSoftmax, DynamicConv1dTBC, GehringLinear,
                          LightweightConv1dTBC, MultiHeadAttention)
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.utils import eval_str_list, fill_with_neg_inf, softmax

from .decoder_base import Decoder, DecoderLayer


@Decoder.register('dynamic_conv_decoder_faces_objects')
class DynamicConvFacesObjectsDecoder(Decoder):
    def __init__(self, vocab, embedder: TextFieldEmbedder, max_target_positions, dropout,
                 share_decoder_input_output_embed,
                 decoder_output_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 decoder_kernel_size_list, adaptive_softmax_cutoff=None,
                 tie_adaptive_weights=False, adaptive_softmax_dropout=0,
                 tie_adaptive_proj=False, adaptive_softmax_factor=0, decoder_layers=6,
                 final_norm=True, padding_idx=0, namespace='target_tokens',
                 vocab_size=None, section_attn=False, swap=False):
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
                                    swap,
                                    kernel_size=decoder_kernel_size_list[i])
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

    def forward(self, prev_target, contexts, incremental_state=None,
                use_layers=None, **kwargs):
        # embed tokens and positions
        X = self.embedder(prev_target, incremental_state=incremental_state)

        # if incremental_state is not None:
        #     X = X[:, -1:]

        if self.project_in_dim is not None:
            X = self.project_in_dim(X)

        X = F.dropout(X, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        X = X.transpose(0, 1)
        attns = []

        inner_states = [X]

        # decoder layers
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(
                    X,
                    contexts,
                    incremental_state,
                )
                inner_states.append(X)
            attns.append(attn)

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

        return X, {'attn': attns, 'inner_states': inner_states}

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


@DecoderLayer.register('dynamic_conv_faces_objects')
class DynamicConvDecoderLayer(DecoderLayer):
    def __init__(self, decoder_embed_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 swap, kernel_size=0):
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

        self.context_attns = nn.ModuleDict()
        self.context_attn_lns = nn.ModuleDict()
        C = 2048

        self.context_attns['image'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=C, vdim=C,
            dropout=attention_dropout)
        self.context_attn_lns['image'] = nn.LayerNorm(self.embed_dim)

        self.context_attns['article'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=1024, vdim=1024,
            dropout=attention_dropout)
        self.context_attn_lns['article'] = nn.LayerNorm(self.embed_dim)

        self.context_attns['faces'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=512, vdim=512,
            dropout=attention_dropout)
        self.context_attn_lns['faces'] = nn.LayerNorm(self.embed_dim)

        self.context_attns['obj'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=2048, vdim=2048,
            dropout=attention_dropout)
        self.context_attn_lns['obj'] = nn.LayerNorm(self.embed_dim)

        context_size = self.embed_dim * 4

        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True
        self.swap = swap

    def forward(self, X, contexts, incremental_state):
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

        attns = {}
        X_contexts = []

        # Image attention
        residual = X
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X, before=True)
        X_image, attn = self.context_attns['image'](
            query=X_image,
            key=contexts['image'],
            value=contexts['image'],
            key_padding_mask=contexts['image_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_image = F.dropout(X_image, p=self.dropout, training=self.training)
        X_image = residual + X_image
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X_image, after=True)
        X_contexts.append(X_image)
        attns['image'] = attn.cpu().detach().numpy()

        # Article attention
        residual = X
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X, before=True)
        X_article, attn = self.context_attns['article'](
            query=X_article,
            key=contexts['article'],
            value=contexts['article'],
            key_padding_mask=contexts['article_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_article = F.dropout(X_article, p=self.dropout,
                              training=self.training)
        X_article = residual + X_article
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X_article, after=True)
        X_contexts.append(X_article)
        attns['article'] = attn.cpu().detach().numpy()

        # Face attention
        residual = X
        X_faces = self.maybe_layer_norm(
            self.context_attn_lns['faces'], X, before=True)
        X_faces, attn = self.context_attns['faces'](
            query=X_faces,
            key=contexts['faces'],
            value=contexts['faces'],
            key_padding_mask=contexts['faces_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_faces = F.dropout(X_faces, p=self.dropout,
                            training=self.training)
        X_faces = residual + X_faces
        X_faces = self.maybe_layer_norm(
            self.context_attn_lns['faces'], X_faces, after=True)
        X_contexts.append(X_faces)
        attns['faces'] = attn.cpu().detach().numpy()

        # Object attention
        residual = X
        X_objs = self.maybe_layer_norm(
            self.context_attn_lns['obj'], X, before=True)
        X_objs, attn = self.context_attns['obj'](
            query=X_objs,
            key=contexts['obj'],
            value=contexts['obj'],
            key_padding_mask=contexts['obj_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_objs = F.dropout(X_objs, p=self.dropout,
                           training=self.training)
        X_objs = residual + X_objs
        X_objs = self.maybe_layer_norm(
            self.context_attn_lns['obj'], X_objs, after=True)
        X_contexts.append(X_objs)
        attns['obj'] = attn.cpu().detach().numpy()

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
        return X, attns

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
