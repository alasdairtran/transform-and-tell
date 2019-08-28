import math
import operator
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
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sort_batch_by_length
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.rouge.rouge import Rouge
from pytorch_transformers.modeling_roberta import RobertaModel
from pytorch_transformers.modeling_utils import SequenceSummary

from .resnet import resnext101_32x16d_wsl

LSTM = _Seq2SeqWrapper(nn.LSTM)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@dataclass
class SequenceSummaryConfig:
    hidden_size: int
    summary_type: str = 'last'
    summary_use_proj: bool = True
    summary_proj_to_labels: bool = False
    summary_activation: str = 'tanh'
    summary_first_dropout: float = 0.1
    summary_last_dropout: float = 0.1


class Attention(nn.Module):
    """Attention Network.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    """

    def __init__(self, n_channels, hidden_size, attention_dim):
        """
        :param n_channels: feature size of encoded images
        :param hidden_size: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super().__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(n_channels, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image, decoder_hidden, softmax=False):
        # image.shape == [batch_size, n_pixels, n_channels]
        # decoder_hidden.shape == [batch_size, hidden_size]

        attn_1 = self.encoder_att(image)
        # attn_1 = [batch_size, n_pixels, attention_size]

        attn_2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        # attn_2 = [batch_size, 1, attention_size]

        attn = torch.cat([attn_1, attn_2.expand_as(attn_1)], dim=2)

        attn = self.full_att(gelu(attn)).squeeze(2)
        # attn.shape == [batch_size, num_pixels]

        if softmax:
            return None, F.softmax(attn, dim=-1)

        alpha = self.softmax(attn)
        # alpha.shape == [batch_size, num_pixels]

        attended_image = (image * alpha.unsqueeze(2)).sum(dim=1)
        # attended_image.shape = [batch_size, n_channels]

        return attended_image, alpha


class ArticleAttention(nn.Module):
    """Article Attention Network.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    """

    def __init__(self, embed_size, hidden_size, attention_dim, pool_type='lstm'):
        """
        :param embed_size: feature size of encoded images
        :param hidden_size: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super().__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(embed_size, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim * 2, 1)

        self.section_attention = Attention(
            embed_size, hidden_size, attention_dim)

        self.pool_type = pool_type
        if pool_type == 'lstm':
            self.pooler = LSTM(input_size=embed_size, hidden_size=embed_size,
                               bidirectional=False, bias=False)
        elif pool_type in ['first', 'mean']:
            self.pooler = nn.Linear(embed_size, embed_size)
        elif pool_type != 'none':
            raise ValueError(f'Unknown pool type: {pool_type}')

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        warnings.filterwarnings(
            "ignore", message="RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().")

    def forward(self, article, article_mask, decoder_hidden, section_mask):
        # article.shape == [batch_size, n_sections, seq_len, embed_size]
        # article_mask.shape == [batch_size, n_sections, seq_len]
        # decoder_hidden.shape == [batch_size, hidden_size]

        B, G, S, E = article.shape

        # Pool each section of the article
        if self.pool_type == 'lstm':
            flattned_article = article.reshape(B * G, S, E)[section_mask]
            flattened_mask = article_mask.view(B * G, S)[section_mask]
            pooled_article_i = self.pooler(flattned_article, flattened_mask)
            # pooled_article.shape == [n_active_sections, seq_len, hidden_size]

            pooled_article_i = pooled_article_i[:, -1]
            # pooled_article.shape == [n_active_sections, hidden_size]

            pooled_article = pooled_article_i.new_zeros(B * G, E)
            pooled_article[section_mask] = pooled_article_i

            pooled_article = pooled_article.view(B, G, E)
            # pooled_article.shape == [batch_size, n_sections, hidden_size]

        elif self.pool_type == 'first':
            pooled_article = article[:, :, 0]
            pooled_article = self.pooler(pooled_article)
            # pooled_article.shape == [batch_size, n_sections, hidden_size]

        elif self.pool_type == 'mean':
            pooled_article = article.mean(dim=2)
            pooled_article = self.pooler(pooled_article)
            # pooled_article.shape == [batch_size, n_sections, hidden_size]

        elif self.pool_type != 'none':
            raise ValueError(f'Unknown pool type: {self.pool_type}')

        if self.pool_type != 'none':
            attn_1 = self.encoder_att(pooled_article)
            # attn_1 = [batch_size, n_sections, attention_size]

            attn_2 = self.decoder_att(decoder_hidden).unsqueeze(1)
            # attn_2 = [batch_size, 1, attention_size]

            attn = torch.cat([attn_1, attn_2.expand_as(attn_1)], dim=2)

            attn = self.full_att(gelu(attn)).squeeze(2)
            # attn.shape == [batch_size, n_sections]

            alpha = self.softmax(attn)
            # alpha.shape == [batch_size, n_sections]

            attended_article = (pooled_article * alpha.unsqueeze(2)).sum(dim=1)
            # attended_article.shape = [batch_size, embed_size]

            # Find the most attended section
            best_idx = alpha.argmax(dim=1)
            # best_idx.shape == [batch_size]

            section = article[torch.arange(B), best_idx]
            # section.shape == [batch_size, seq_len, embed_size]

            attended_section, alpha_2 = self.section_attention(
                section, decoder_hidden)

            return attended_article, attended_section, alpha, alpha_2

        else:
            section = article.reshape(B, G * S, E)[:, :512]
            attended_section, alpha_2 = self.section_attention(
                section, decoder_hidden)
            return None, attended_section, None, alpha_2


@Model.register("pointer")
class PointerModel(Model):
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
                 topk: int = 1,
                 pool_type: str = 'lstm',
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        n_channels = 2048
        text_embed_size = 1024
        self.index = index
        self.namespace = namespace
        self.resnet = resnext101_32x16d_wsl()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.attention = Attention(n_channels, hidden_size, attention_dim)
        self.use_context = use_context
        self.topk = topk
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode

        self.attention_copy = Attention(
            text_embed_size, hidden_size, attention_dim)

        # Projection so that image and text embeds have same dimension
        # self.image_proj = nn.Linear(2048, 384)
        # self.text_proj = nn.Linear(1024, 384)

        # Linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(n_channels, hidden_size)

        # Linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(n_channels, hidden_size)

        # Linear layer to create a sigmoid-activation gate
        self.f_beta = nn.Linear(hidden_size, n_channels)

        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Linear layer to decide whether to copy or not
        self.fc_copy = nn.Linear(hidden_size, 2)

        if use_context:
            self.article_attention = ArticleAttention(
                text_embed_size, hidden_size, attention_dim, pool_type)
            self.f_beta_2 = nn.Linear(hidden_size, text_embed_size)
            self.f_beta_3 = nn.Linear(hidden_size, text_embed_size)
            if pool_type == 'none':
                input_size = n_channels + text_embed_size * 2
            else:
                input_size = n_channels + text_embed_size * 3
        else:
            input_size = n_channels + text_embed_size

        self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)

        initializer(self)

        # Initialize the weight with first layer of BERT
        self.fc.weight.data.copy_(
            self.roberta.model.decoder.sentence_encoder.embed_tokens.weight)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        caption_ids, image_embeds, caption_embeds, context_mask, context_ids, context_embeds, caption_lens, sort_index = self._forward(
            context, image, caption, metadata)

        metadata = list(np.array(metadata)[sort_index.cpu().numpy()])
        B, P, C = image_embeds.shape
        B, G, S, E = context_embeds.shape

        # We won't decode at the </s> position, since we've finished
        # generating as soon as we generate </s>.
        decode_lens = (caption_lens - 1).tolist()

        # Initialize LSTM state
        h, c = self.init_hidden_state(image_embeds)

        # We won't decode at the </s> position, since we've finished
        # generating as soon as we generate </s>.
        decode_lens = (caption_lens - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        V = self.vocab.get_vocab_size(self.namespace)
        L = max(decode_lens)
        logits = image_embeds.new_zeros(B, L, V)
        logits_decision = image_embeds.new_zeros(B, L, 2)
        copy_probs = image_embeds.new_zeros(B, L, G * S)
        alphas = image_embeds.new_zeros(B, L, P)

        targets = caption_ids[:, 1:]

        # At each time-step, decode by attention-weighing the encoder's output
        # based on the decoder's previous hidden state output then generate a
        # new word in the decoder with the previous word and the attention
        # weighted encoding
        for t in range(L):
            batch_size_t = sum([length > t for length in decode_lens])
            image_embeds_t = image_embeds[:batch_size_t]
            caption_embeds_t = caption_embeds[:batch_size_t]
            h_t = h[:batch_size_t]
            c_t = c[:batch_size_t]

            attended_image, alpha = self.attention(image_embeds_t, h_t)
            # attended_image.shape = [batch_size_t, n_channels]
            # alpha.shape == [batch_size_t, num_pixels]

            gate = torch.sigmoid(self.f_beta(h_t))
            # gate.shape == [batch_size_t, n_channels]

            attended_image = gate * attended_image
            # attended_image.shape = [batch_size_t, n_channels]

            # The input to the RNN cell is a concatenation of
            #   1) The word from the previous step of the caption
            #   2) The attended image
            prev_word = caption_embeds_t[:, t, :]

            if self.use_context:
                context_embeds_t = context_embeds[:batch_size_t]
                context_mask_t = context_mask[:batch_size_t]
                context_ids_t = context_ids.view(
                    B, G, -1)[:batch_size_t].view(batch_size_t * G, -1)
                section_mask_t = ~(context_ids_t == self.padding_idx).all(1)
                attended_article, attended_section, _, _ = self.article_attention(
                    context_embeds_t, context_mask_t, h_t, section_mask_t)

                gate_3 = torch.sigmoid(self.f_beta_3(h_t))
                attended_section = gate_3 * attended_section

                rnn_input = torch.cat(
                    [prev_word, attended_image, attended_section], dim=1)

                if attended_article is not None:
                    gate_2 = torch.sigmoid(self.f_beta_2(h_t))
                    attended_article = gate_2 * attended_article

                    rnn_input = torch.cat([rnn_input, attended_article], dim=1)

            else:
                rnn_input = torch.cat([prev_word, attended_image], dim=1)

            # Feed through the RNN cell
            h, c = self.rnn_cell(rnn_input, (h_t, c_t))

            # Project onto the vocabulary
            logits_t = self.fc(self.dropout(h))
            # logits_t.shape = [batch_size_t, vocab_size]

            logits_decision_t = self.fc_copy(self.dropout(h))
            # logits_decision_t.shape = [batch_size_t, 2]

            logits[:batch_size_t, t, :] = logits_t
            logits_decision[:batch_size_t, t, :] = logits_decision_t
            alphas[:batch_size_t, t, :] = alpha

            context_embeds_t = context_embeds_t.view(batch_size_t, G * S, E)
            _, copy_probs_t = self.attention_copy(
                context_embeds_t, h, softmax=True)
            # copy_probs.shape == [batch_size_t, G * S]

            copy_probs[:batch_size_t, t, :] = copy_probs_t

        gen_probs = F.softmax(logits, dim=-1)
        # gen_probs.shape == [batch_size, seq_len, vocab_size]

        decision_probs = F.softmax(logits_decision, dim=-1)
        # decision_probs.shape == [batch_size, seq_len, 2]

        # copy_probs.shape == [batch_size, seq_len, src_len]

        # the second probability is the generation prob
        gen_probs = gen_probs * decision_probs[:, :, 1:]
        # gen_probs.shape == [batch_size, seq_len, vocab_size]

        copy_probs = copy_probs * decision_probs[:, :, :1]
        # copy_probs.shape == [batch_size, seq_len, src_len]

        projected_copy_probs = gen_probs.new_zeros(gen_probs.shape)
        # projected_copy_probs.shape == [batch_size, seq_len, vocab_size]

        context_ids = context_ids.view(B, -1)
        # context_ids = [batch_size, src_len]

        index = context_ids.unsqueeze(1)
        # context_ids = [batch_size, 1, src_len]

        index = index.expand_as(copy_probs)
        # context_ids = [batch_size, seq_len, src_len]

        projected_copy_probs.scatter_add_(2, index, copy_probs)
        # projected_copy_probs.shape == [batch_size, seq_len, vocab_size]

        probs = projected_copy_probs + gen_probs

        probs = probs.view(-1, V)

        text_mask = targets.ne(self.padding_idx)
        targets = targets[text_mask].contiguous().view(-1, 1)
        probs = probs[text_mask.view(-1)]

        loss = -torch.log(probs.gather(dim=-1, index=targets)).mean()

        self.n_batches += 1
        self.n_samples += B

        # During evaluation, we will generate a caption and compute BLEU, etc.
        if not self.training and self.evaluate_mode:
            gen_dict = self._generate(caption_ids, image_embeds,
                                      caption_embeds, context_embeds, context_mask, context_ids)
            gen_texts = gen_dict['generated_texts']
            captions = [m['caption'] for m in metadata]

            # Remove punctuation
            gen_texts = [re.sub(r'[^\w\s]', '', t) for t in gen_texts]
            captions = [re.sub(r'[^\w\s]', '', t) for t in captions]

            for gen, ref in zip(gen_texts, captions):
                bleu_scorer = BleuScorer(n=4)
                bleu_scorer += (gen, [ref])
                score, _ = bleu_scorer.compute_score(option='closest')
                self.sample_history['bleu-1'] += score[0] * 100
                self.sample_history['bleu-2'] += score[1] * 100
                self.sample_history['bleu-3'] += score[2] * 100
                self.sample_history['bleu-4'] += score[3] * 100

                rogue_scorer = Rouge()
                score = rogue_scorer.calc_score([gen], [ref])
                self.sample_history['rogue'] += score * 100

        loss = loss / math.log(2)
        output_dict = {'loss': loss}
        return output_dict

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        caption_ids, image_embeds, caption_embeds, context_mask, context_ids, context_embeds, _, sort_index = self._forward(
            context, image, caption, metadata)
        metadata = list(np.array(metadata)[sort_index.cpu().numpy()])
        output_dict = self._generate(caption_ids, image_embeds,
                                     caption_embeds, context_embeds, context_mask, context_ids)
        output_dict['captions'] = [m['caption'] for m in metadata]
        output_dict['web_url'] = [m['web_url'] for m in metadata]
        return output_dict

    def _forward(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: List[Dict[str, Any]]):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        # STEP 1: Embed the caption
        # Sort the caption by decreasing lengths (save computation later on)
        caption_ids = caption[self.index]
        # caption_ids.shape == [batch_size, seq_len]

        # Assume the padding token ID is 1
        caption_mask = caption_ids != self.padding_idx
        # caption_ids.shape == [batch_size, seq_len]

        caption_lens = caption_mask.sum(dim=1)
        # caption_len.shape == [batch_size]

        caption_ids, caption_lens, _, sort_index = sort_batch_by_length(
            caption_ids, caption_lens)

        caption_embeds = self.roberta.extract_features(
            caption_ids, return_all_hiddens=True)[0]
        # caption_embeds.shape == [batch_size, seq_len, 1024]

        # STEP 2: Embed the image
        # image.shape == [batch_size, 3, 224, 224]
        image = self.resnet(image)
        # image.shape == [batch_size, 2048, 7, 7]

        image_embeds = image.permute(0, 2, 3, 1)
        # image_embeds.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = image_embeds.shape
        P = H * W  # number of pixels
        image_embeds = image_embeds.view(B, P, C)
        image_embeds = image_embeds[sort_index]
        # image_embeds.shape == [batch_size, 49, 2048]

        # image_embeds = self.image_proj(image_embeds)
        # image_embeds.shape == [batch_size, 49, 1024]

        if self.use_context:
            context_ids = context[self.index]
            if len(context_ids.shape) == 2:
                context_ids = context_ids.unsqueeze(1)
            context_mask = context_ids != self.padding_idx

            B, G, S = context_ids.shape
            context_ids = context_ids.view(B * G, S)
            # context_ids.shape == [batch_size * n_sections, seq_len]

            context_embeds = self.roberta.extract_features(context_ids)
            context_embeds = context_embeds.view(B, G, S, -1)
            # context_embeds.shape == [batch_size, n_sections, seq_len, 1024]

            context_embeds = context_embeds[sort_index]
            context_ids = context_ids.view(B, G, S)[sort_index].view(B * G, S)
            context_mask = context_mask[sort_index]
        else:
            context_embeds = None
            context_ids = None

        return caption_ids, image_embeds, caption_embeds, context_mask, context_ids, context_embeds, caption_lens, sort_index

    def _generate(self, caption_ids, image_embeds,
                  caption_embeds, context_embeds, context_mask, context_ids):

        # Initialize LSTM state
        h, c = self.init_hidden_state(image_embeds)

        # Create tensors to hold word prediction scores and alphas
        V = self.vocab.get_vocab_size(self.namespace)
        max_len = 32
        B = image_embeds.shape[0]

        # Assume <s> is 0, <pad> is 1, and </s> is 2.
        # We won't store the first <s> in generated
        generated = caption_ids.new_ones(B, max_len)
        eos = 2
        is_end = generated[:, 0] == eos

        BG, S = context_ids.shape
        context_ids = context_ids.view(B, -1)

        # At each time-step, decode by attention-weighing the encoder's output
        # based on the decoder's previous hidden state output then generate a
        # new word in the decoder with the previous word and the attention
        # weighted encoding.
        for t in range(max_len):
            attended_image, alpha = self.attention(image_embeds, h)
            # attended_image.shape = [batch_size_t, n_channels]
            # alpha.shape == [batch_size_t, num_pixels]

            gate = torch.sigmoid(self.f_beta(h))
            # gate.shape == [batch_size_t, n_channels]

            attended_image = gate * attended_image
            # attended_image.shape = [batch_size_t, n_channels]

            # The input to the RNN cell is a concatenation of
            #   1) The word from the previous step of the caption
            #   2) The attended image
            if t == 0:
                prev_word = caption_embeds[:, 0, :]
                generated[:, 0] = caption_ids[:, 0]
            else:
                top_probs, top_indices = probs_t.topk(self.topk, dim=-1)
                # top_indices.shape == [batch_size_t, k]

                idx = torch.multinomial(top_probs, 1)
                # idx.shape == [batch_size, 1]

                word_idx = torch.gather(top_indices, dim=1, index=idx)
                word_idx = word_idx.squeeze(-1)
                # word_idx.shape == [batch_size]

                generated[~is_end, t] = word_idx[~is_end]

                # Once we've reached </s>, is_end will become and remain True
                is_end |= word_idx == eos
                if is_end.sum().item() == len(is_end):
                    break

                prev_word = self.roberta.extract_features(
                    generated[:, :t+1], return_all_hiddens=True)
                prev_word = prev_word[0][:, -1]
                # prev_word.shape == [batch_size, embed_size]

            if self.use_context:
                section_mask = ~(context_ids == self.padding_idx).all(1)
                attended_article, attended_section, _, _ = self.article_attention(
                    context_embeds, context_mask, h, section_mask)

                gate_3 = torch.sigmoid(self.f_beta_3(h))
                attended_section = gate_3 * attended_section
                rnn_input = torch.cat(
                    [prev_word, attended_image, attended_section], dim=1)

                if attended_article is not None:
                    gate_2 = torch.sigmoid(self.f_beta_2(h))
                    attended_article = gate_2 * attended_article
                    rnn_input = torch.cat([rnn_input, attended_article], dim=1)

            else:
                rnn_input = torch.cat([prev_word, attended_image], dim=1)

            # Feed through the RNN cell
            h, c = self.rnn_cell(rnn_input, (h, c))

            # Project onto the vocabulary
            logits_t = self.fc(self.dropout(h))
            # logits_t.shape = [batch_size, vocab_size]

            gen_probs_t = F.softmax(logits_t, dim=-1)
            # gen_probs_t.shape = [batch_size, vocab_size]

            logits_decision_t = self.fc_copy(self.dropout(h))
            # logits_decision_t.shape = [batch_size, 2]

            decision_probs_t = F.softmax(logits_decision_t, dim=-1)
            # decision_probs.shape == [batch_size, 2]

            B, G, S, E = context_embeds.shape
            context_embeds_t = context_embeds.view(B, G * S, E)
            _, copy_probs_t = self.attention_copy(
                context_embeds_t, h, softmax=True)
            # copy_probs.shape == [batch_size, G * S]

            copy_probs_t = copy_probs_t * decision_probs_t[:, :1]

            gen_probs_t = gen_probs_t * decision_probs_t[:, 1:]

            projected_copy_probs = gen_probs_t.new_zeros(gen_probs_t.shape)
            # projected_copy_probs.shape == [batch_size, vocab_size]

            index = context_ids.view(B, -1)
            # context_ids = [batch_size, src_len]

            projected_copy_probs.scatter_add_(1, index, copy_probs_t)
            # projected_copy_probs.shape == [batch_size, vocab_size]

            probs_t = projected_copy_probs + gen_probs_t
            # probs_t.shape == [batch_size, vocab_size]

        self.n_batches += 1
        self.n_samples += B

        gen_indices = generated[:, :t].cpu()
        gen_texts = [self.roberta.decode(x[x != 1]) for x in gen_indices]

        return {
            'generated_indices': gen_indices,
            'generated_texts': gen_texts,
        }

    def init_hidden_state(self, image_embeds):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # image_embeds.shape == [batch_size, n_pixels, n_channels]
        mean_image_embeds = image_embeds.mean(dim=1)
        # mean_image_embeds.shape == [batch_size, n_pixels, n_channels]

        h = self.init_h(mean_image_embeds)  # (batch_size, decoder_dim)
        c = self.init_c(mean_image_embeds)
        return h, c

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
