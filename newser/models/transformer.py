import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sort_batch_by_length
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.rouge.rouge import Rouge
from pytorch_transformers.modeling_roberta import RobertaModel
from pytorch_transformers.modeling_utils import SequenceSummary

from newser.modules import GehringLinear
from newser.modules.criteria import Criterion

from .decoder import Decoder
from .resnet import resnext101_32x16d_wsl

LSTM = _Seq2SeqWrapper(nn.LSTM)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_embeds, n_pos, n_heads, n_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.ln_1, self.ln_2 = nn.ModuleList(), nn.ModuleList()

        for _ in range(n_layers):
            self.attentions.append(nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, X_caption, X_image, X_article):
        h = self.dropout(X_caption)

        # In attn_mask, the upper triangle excluding the diagonal, will be
        # set to -inf. So we can attend to ourselves and the past.
        attn_mask = torch.full((len(X_caption), len(X_caption)), -float('inf'),
                               device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        for ln_1, attn, ln_2, ff in zip(self.ln_1, self.attentions, self.ln_2, self.feed_forwards):
            h = ln_1(h)
            X_caption, _ = attn(
                h, h, h, attn_mask=attn_mask, need_weights=False)
            X_caption = self.dropout(X_caption)
            h = X_caption + h

            h = ln_2(h)
            h = ff(h)
            h = self.dropout(X_caption)
            h = X_caption + h

        return h


@dataclass
class SequenceSummaryConfig:
    hidden_size: int
    summary_type: str = 'last'
    summary_use_proj: bool = True
    summary_proj_to_labels: bool = False
    summary_activation: str = 'tanh'
    summary_first_dropout: float = 0.1
    summary_last_dropout: float = 0.1


@Model.register("transformer")
class TransformerModel(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 criterion: Criterion,
                 evaluate_mode: bool = False,
                 attention_dim: int = 1024,
                 hidden_size: int = 1024,
                 dropout: float = 0.1,
                 vocab_size: int = 50264,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 index: str = 'roberta',
                 padding_value: int = 1,
                 use_context: bool = True,
                 sampling_topk: int = 1,
                 sampling_temp: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.decoder = decoder
        self.criterion = criterion

        self.index = index
        self.namespace = namespace
        self.resnet = resnext101_32x16d_wsl()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)
        self.image_proj = GehringLinear(2048, 1024, bias=False)

        initializer(self)

        # Initialize the weight with first layer of BERT
        # self.fc.weight.data.copy_(
        #     self.roberta.model.decoder.sentence_encoder.embed_tokens.weight)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        _, target_ids, contexts, context_masks = self._forward(
            context, image, caption)
        decoder_out = self.decoder(caption, contexts, context_masks)

        # Assume we're using adaptive loss
        loss, sample_size = self.criterion(
            self.decoder.adaptive_softmax, decoder_out, target_ids)

        loss = loss / math.log(2)

        output_dict = {
            'loss': loss / sample_size,
            'sample_size': sample_size,
        }

        return output_dict

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        caption_ids, _, contexts, context_masks = self._forward(
            context, image, caption)

        _, gen_ids = self._generate(caption_ids, contexts, context_masks)

        gen_ids = gen_ids.cpu()
        gen_texts = [self.roberta.decode(
            x[x != self.padding_idx]) for x in gen_ids]

        output_dict = {
            'generated_indices': gen_ids,
            'generated_texts': gen_texts,
            'captions': [m['caption'] for m in metadata],
            'web_url': [m['web_url'] for m in metadata],
        }

        return output_dict

    def _forward(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor]):

        # We assume that the first token in target is the <s> token. We
        # shall use it to seed the decoder. Here decoder_target is simply
        # decoder_input but shifted to the right by one step.
        caption_ids = caption[self.index]
        target_ids = torch.zeros_like(caption_ids)
        target_ids[:, :-1] = caption_ids[:, 1:]

        # The final token is not used as input to the decoder, since otherwise
        # we'll be predicting the <pad> token.
        caption_ids = caption_ids[:, :-1]
        target_ids = target_ids[:, :-1]
        caption[self.index] = caption_ids

        # Embed the image
        X_image = self.resnet(image)
        # X_image.shape == [batch_size, 2048, 7, 7]

        X_image = X_image.permute(0, 2, 3, 1)
        # X_image.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = X_image.shape
        P = H * W  # number of pixels
        X_image = X_image.view(B, P, C)
        # X_image.shape == [batch_size, 49, 2048]

        X_image = self.image_proj(X_image)
        # X_image.shape == [batch_size, 49, 1024]

        # Embed article
        article_ids = context[self.index][:, :, :128]
        # article_ids.shape == [batch_size, n_sections, seq_len]
        article_ids = article_ids[:, :2]

        article_padding_mask = article_ids == self.padding_idx

        B, G, S = article_ids.shape
        article_ids = article_ids.reshape(B * G, S)
        # article_ids.shape == [batch_size * n_sections, seq_len]

        X_article = self.roberta.extract_features(article_ids)
        # X_article.shape == [batch_size * n_sections, seq_len, embed_size]

        X_article = X_article.view(B, G, S, -1)
        # X_article.shape == [batch_size, n_sections, seq_len, embed_size]

        # First try: article is only the title and first paragraph
        X_article = X_article.reshape(B, G * S, -1)

        # Create padding mask (1 corresponds to the padding index)
        image_padding_mask = X_image.new_zeros(B, P).bool()
        article_padding_mask = article_padding_mask.view(B, G * S).bool()

        contexts = [X_image, X_article]
        context_masks = [image_padding_mask, article_padding_mask]

        return caption_ids, target_ids, contexts, context_masks

    def _generate(self, caption_ids, contexts, context_masks):
        incremental_state: Dict[str, Any] = {}
        seed_input = caption_ids[:, 0:1]
        log_prob_list = []
        index_path_list = [seed_input]
        eos = 2
        active_idx = seed_input[:, -1] != eos
        full_active_idx = active_idx
        gen_len = 100
        B = caption_ids.shape[0]

        for i in range(gen_len):
            if i == 0:
                prev_target = {self.index: seed_input}
            else:
                prev_target = {self.index: seed_input[:, -1:]}

            self.decoder.filter_incremental_state(
                incremental_state, active_idx)

            contexts_i = [ctx[full_active_idx] for ctx in contexts]
            context_masks_i = [mask[full_active_idx] for mask in context_masks]
            decoder_out = self.decoder(
                prev_target,
                contexts_i,
                context_masks_i,
                incremental_state=incremental_state)

            # We're only interested in the current final word
            decoder_out = (decoder_out[0][:, -1:], None)

            lprobs = self.decoder.get_normalized_probs(
                decoder_out, log_probs=True)
            # lprobs.shape == [batch_size, 1, vocab_size]

            lprobs = lprobs.squeeze(1)
            # lprobs.shape == [batch_size, vocab_size]

            topk_lprobs, topk_indices = lprobs.topk(self.sampling_topk)
            topk_lprobs = topk_lprobs.div_(self.sampling_temp)
            # topk_lprobs.shape == [batch_size, topk]

            # Take a random sample from those top k
            topk_probs = topk_lprobs.exp()
            sampled_index = torch.multinomial(topk_probs, num_samples=1)
            # sampled_index.shape == [batch_size, 1]

            selected_lprob = topk_lprobs.gather(
                dim=-1, index=sampled_index)
            # selected_prob.shape == [batch_size, 1]

            selected_index = topk_indices.gather(
                dim=-1, index=sampled_index)
            # selected_index.shape == [batch_size, 1]

            log_prob = selected_lprob.new_zeros(B, 1)
            log_prob[full_active_idx] = selected_lprob

            index_path = selected_index.new_zeros(B, 1)
            index_path.fill_(self.padding_idx)
            index_path[full_active_idx] = selected_index

            log_prob_list.append(log_prob)
            index_path_list.append(index_path)

            seed_input = torch.cat([seed_input, selected_index], dim=-1)

            is_eos = selected_index.squeeze(-1) == eos
            active_idx = ~is_eos

            full_active_idx[full_active_idx.nonzero()[~active_idx]] = 0

            seed_input = seed_input[active_idx]

            if active_idx.sum().item() == 0:
                break

        log_probs = torch.cat(log_prob_list, dim=-1)
        # log_probs.shape == [batch_size * beam_size, generate_len]

        token_ids = torch.cat(index_path_list, dim=-1)
        # token_ids.shape == [batch_size * beam_size, generate_len]

        return log_probs, token_ids

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics['_n_batches'] = self.n_batches
        metrics['_n_samples'] = self.n_samples

        for key, value in self.sample_history.items():
            metrics[key] = value / self.n_samples

        if reset:
            self.n_batches = 0
            self.n_samples = 0
            self.sample_history: Dict[str, float] = defaultdict(float)

        return metrics
