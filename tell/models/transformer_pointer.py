import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from tell.modules import (GehringLinear, LoadStateDictWithPrefix,
                            SelfAttention, multi_head_attention_score_forward)
from tell.modules.criteria import Criterion

from .decoder_flattened import Decoder
from .resnet import resnet152

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

LSTM = _Seq2SeqWrapper(nn.LSTM)


@Model.register("transformer_pointer")
class TransformerPointerModel(LoadStateDictWithPrefix, Model):
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
                 weigh_bert: bool = False,
                 model_path: str = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.decoder = decoder
        self.criterion = criterion

        self.index = index
        self.namespace = namespace
        self.resnet = resnet152()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp
        self.weigh_bert = weigh_bert
        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

            self.bert_weight_2 = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight_2)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)
        self.batch_history: Dict[str, float] = defaultdict(float)

        self.entity_fc = GehringLinear(1024, 2)
        self.entity_loss = nn.CrossEntropyLoss(ignore_index=-1)
        # self.copy_loss = nn.CrossEntropyLoss(ignore_index=padding_value)

        # Copy-related modules
        self.in_proj_weight = nn.Parameter(torch.empty(2 * 1024, 1024))
        self.in_proj_bias = nn.Parameter(torch.empty(2 * 1024))
        self.out_proj = GehringLinear(1024, 1024, bias=True)
        self.bias_k = nn.Parameter(torch.empty(1, 1, 1024))
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        xavier_normal_(self.bias_k)

        # Entity-related modules
        self.entity_attn = SelfAttention(
            out_channels=1024, embed_dim=1024, num_heads=16, gated=True)

        initializer(self)
        self.vocab_size = vocab_size

        if model_path is not None:
            logger.info(f'Recovering weights from {model_path}.')
            model_state = torch.load(model_path)
            self.load_state_dict(model_state)
        # Initialize the weight with first layer of BERT
        # self.fc.weight.data.copy_(
        #     self.roberta.model.decoder.sentence_encoder.embed_tokens.weight)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                face_embeds: torch.Tensor,
                metadata: List[Dict[str, Any]],
                names=None) -> Dict[str, torch.Tensor]:

        caption_ids, target_ids, contexts, X_sections_hiddens, article_padding_mask = self._forward(
            context, image, caption, face_embeds)
        decoder_out = self.decoder(caption, contexts)

        # Assume we're using adaptive loss
        gen_loss, sample_size = self.criterion(
            self.decoder.adaptive_softmax, decoder_out, target_ids)

        entity_loss, copy_loss = self.pointer_loss(
            decoder_out, context, caption, target_ids, X_sections_hiddens, article_padding_mask)

        gen_loss = gen_loss / sample_size / math.log(2)
        entity_loss = entity_loss / math.log(2)
        copy_loss = copy_loss / math.log(2)

        loss = entity_loss + copy_loss

        if self.training and not loss.requires_grad:
            loss = None

        if not torch.isnan(gen_loss):
            self.batch_history['gen_loss'] += gen_loss.item()
        if not torch.isnan(entity_loss):
            self.batch_history['entity_loss'] += entity_loss.item()
        if not torch.isnan(copy_loss):
            self.batch_history['copy_loss'] += copy_loss.item()

        output_dict = {
            'loss': loss,
            'sample_size': sample_size,
        }

        # During evaluation, we will generate a caption and compute BLEU, etc.
        if not self.training and self.evaluate_mode:
            log_probs, copy_probs, should_copy_mask, gen_ids = self._generate(
                caption_ids, contexts, X_sections_hiddens, article_padding_mask, context)
            gen_texts = [self.roberta.decode(x[x > 1]) for x in gen_ids.cpu()]
            captions = [m['caption'] for m in metadata]

            copied_texts = [self.roberta.decode(x[should_copy_mask[i]])
                            for i, x in enumerate(gen_ids.cpu())]

            output_dict['captions'] = captions
            output_dict['generations'] = gen_texts
            output_dict['metadata'] = metadata
            output_dict['copied_texts'] = copied_texts

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

                # rogue_scorer = Rouge()
                # score = rogue_scorer.calc_score([gen], [ref])
                # self.sample_history['rogue'] += score * 100

        self.n_samples += caption_ids.shape[0]
        self.n_batches += 1

        return output_dict

    def pointer_loss(self, decoder_out, context, caption, caption_targets,
                     X_sections_hiddens, article_padding_mask):
        X = decoder_out[0]
        # X.shape == [batch_size, target_len, embed_size]

        caption_copy_masks = caption[f'{self.index}_copy_masks']
        caption_copy_masks = caption_copy_masks[:, 1:]
        # caption_copy_masks.shape == [batch_size, target_len]

        if not caption_copy_masks[caption_copy_masks >= 1].bool().any():
            return torch.tensor(0.0).to(X.device), torch.tensor(0.0).to(X.device)

        context_copy_masks = context[f'{self.index}_copy_masks']
        # context_copy_masks.shape == [batch_size, source_len]

        if self.weigh_bert:
            X_article = torch.stack(X_sections_hiddens, dim=2)
            # X_article.shape == [batch_size, seq_len, 13, embed_size]

            weight = F.softmax(self.bert_weight_2, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            # weight.shape == [1, 1, 13, 1]

            X_article = (X_article * weight).sum(dim=2)
            # X_article.shape == [batch_size, seq_len, embed_size]

        else:
            X_article = X_sections_hiddens[-1]
            # X_article.shape == [batch_size, seq_len, embed_size]

        X = X.transpose(0, 1)
        # X.shape == [target_len, batch_size, embed_size]

        X_article = X_article.transpose(0, 1)

        X_entity = self.entity_attn(X)
        # X_entity.shape == [target_len, batch_size, embed_size]

        X_entity = X_entity.transpose(0, 1)
        # X_entity.shape == [batch_size, target_len, embed_size]

        entity_logits = self.entity_fc(X_entity)
        # entity_logits.shape == [batch_size, target_len, 2]

        entity_logits = entity_logits.view(-1, 2)
        # entity_logits.shape == [batch_size * target_len, 2]

        targets = caption_copy_masks.clone().reshape(-1)
        targets[targets > 1] = 1
        # targets.shape == [batch_size * target_len]

        entity_loss = self.entity_loss(entity_logits, targets)

        copy_attn = multi_head_attention_score_forward(
            X, X_article, 1024, 16,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, True, 0.1, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=article_padding_mask)
        # copy_attn.shape == [batch_size, target_len, source_len + 2]

        copy_attn = copy_attn[:, :, :-2]
        # copy_attn.shape == [batch_size, target_len, source_len]

        context_copy_masks = context_copy_masks.unsqueeze(1)
        # context_copy_masks.shape == [batch_size, 1, source_len]

        context_copy_masks = context_copy_masks.expand_as(copy_attn)
        # context_copy_masks.shape == [batch_size, target_len, source_len]

        irrelevant_mask = context_copy_masks < 1
        copy_attn[irrelevant_mask] = 0
        # copy_attn.shape == [batch_size, target_len, source_len]

        B, L, S = copy_attn.shape
        copy_probs = copy_attn.new_zeros(B, L, self.vocab_size)
        # copy_probs.shape == [batch_size, target_len, vocab_size]

        context_ids = context[self.index]
        # context_ids.shape == [batch_size, source_len]

        ########################################
        # Second attempt at calculating copy loss
        # First construct the reduced dictionary, containing only tokens
        # mentioned in the context.
        unique_ids = torch.cat([context_ids, caption_targets], dim=1).unique()
        V = len(unique_ids)
        # unique_ids.shape == [reduced_vocab_size]

        # Construct the inverse map of unique_ids
        inverse_unique_ids = unique_ids.new_full([self.vocab_size], -1)
        inverse_unique_ids.index_copy_(
            0, unique_ids, torch.arange(V).to(unique_ids.device))
        # inverse_unique_ids.shape == [vocab_size]
        # e.g. [-1, -1, 0, -1, -1, 1, 2, -1, 3, ....]

        # Next we need to remap the context_ids to the new dictionary.
        new_context_ids = inverse_unique_ids.index_select(
            0, context_ids.reshape(-1))
        # new_context_ids.shape == [batch_size * source_len]

        new_context_ids = new_context_ids.view(B, S)
        new_context_ids = new_context_ids.unsqueeze(1).expand_as(copy_attn)
        # new_context_ids.shape == [batch_size, target_len, source_len]

        new_caption_targets = inverse_unique_ids.index_select(
            0, caption_targets.reshape(-1))
        # new_caption_targets.shape == [batch_size * source_len, 1]

        new_caption_targets = new_caption_targets.reshape(-1, 1)
        # new_caption_targets.shape == [batch_size * n_entity_tokens, 1]

        copy_probs = copy_attn.new_zeros(B, L, V)
        # copy_probs.shape == [batch_size, target_len, reduced_vocab_size]

        copy_probs.scatter_add_(2, new_context_ids, copy_attn)
        copy_lprobs = copy_probs.new_zeros(copy_probs.shape)
        copy_lprobs[copy_probs > 0] = torch.log(copy_probs[copy_probs > 0])
        copy_lprobs = copy_lprobs.view(B * L, V)

        max_index = caption_copy_masks.max().item()
        copy_loss = torch.tensor(0.0).to(X.device)
        for i in range(1, max_index + 1):
            relevant_mask = (caption_copy_masks == i).view(-1)
            new_caption_targets_i = new_caption_targets[relevant_mask]
            # new_caption_targets_i.shape == [batch_size * n_entity_tokens, 1]

            copy_lprobs_i = copy_lprobs[relevant_mask]
            # copy_lprobs_i.shape == [batch_size * n_entity_tokens, reduced_vocab_size]

            copy_loss += -copy_lprobs_i.gather(dim=-1,
                                               index=new_caption_targets_i).mean()

        return entity_loss, copy_loss

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        caption_ids, _, contexts, X_sections_hiddens, article_padding_mask = self._forward(
            context, image, caption)

        log_probs, copy_probs, should_copy_mask, gen_ids = self._generate(
            caption_ids, contexts, X_sections_hiddens, article_padding_mask, context)

        gen_ids = gen_ids.cpu()
        gen_texts = [self.roberta.decode(
            x[x != self.padding_idx]) for x in gen_ids]

        # Get the copied words
        copied_texts = [self.roberta.decode(x[should_copy_mask[i]])
                        for i, x in enumerate(gen_ids)]

        output_dict = {
            'generated_indices': gen_ids,
            'generated_texts': gen_texts,
            'copied_texts': copied_texts,
            'captions': [m['caption'] for m in metadata],
            'web_url': [m['web_url'] for m in metadata],
        }

        return output_dict

    def _forward(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 face_embeds):

        # We assume that the first token in target is the <s> token. We
        # shall use it to seed the decoder. Here decoder_target is simply
        # decoder_input but shifted to the right by one step.
        # For masks, 1 is padding, 2 is normal word, 3 is entity word
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

        article_ids = context[self.index]
        # article_ids.shape == [batch_size, seq_len]

        article_padding_mask = article_ids == self.padding_idx
        # article_padding_mask.shape == [batch_size, seq_len]

        B, S = article_ids.shape

        X_sections_hiddens = self.roberta.extract_features(
            article_ids, return_all_hiddens=True)

        if self.weigh_bert:
            X_article = torch.stack(X_sections_hiddens, dim=2)
            # X_article.shape == [batch_size, seq_len, 13, embed_size]

            weight = F.softmax(self.bert_weight, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            # weight.shape == [1, 1, 13, 1]

            X_article = (X_article * weight).sum(dim=2)
            # X_article.shape == [batch_size, seq_len, embed_size]

        else:
            X_article = X_sections_hiddens[-1]
            # X_article.shape == [batch_size, seq_len, embed_size]

        # Create padding mask (1 corresponds to the padding index)
        image_padding_mask = X_image.new_zeros(B, P).bool()

        # face_embeds.shape == [batch_size, n_faces, 512]
        face_masks = torch.isnan(face_embeds).any(dim=-1)
        face_embeds[face_masks] = 0

        # The quirks of dynamic convolution implementation: The context
        # embedding has dimension [seq_len, batch_size], but the mask has
        # dimension [batch_size, seq_len].
        contexts = {
            'image': X_image.transpose(0, 1),
            'image_mask': image_padding_mask,
            'article': X_article.transpose(0, 1),
            'article_mask': article_padding_mask,
            'sections': None,
            'sections_mask': None,
            'faces': face_embeds.transpose(0, 1),
            'faces_mask': face_masks,
        }

        return caption_ids, target_ids, contexts, X_sections_hiddens, article_padding_mask

    def _generate(self, caption_ids, contexts, X_sections_hiddens, article_padding_mask, context):
        incremental_state: Dict[str, Any] = {}
        seed_input = caption_ids[:, 0:1]
        B = caption_ids.shape[0]
        log_prob_list = []
        copy_prob_list = []
        should_copy_list = [seed_input.new_ones(B, 1).bool()]
        index_path_list = [seed_input]
        eos = 2
        active_idx = seed_input[:, -1] != eos
        full_active_idx = active_idx
        gen_len = 100

        context_copy_masks = context[f'{self.index}_copy_masks']
        # context_copy_masks.shape == [batch_size, source_len]

        if self.weigh_bert:
            X_article = torch.stack(X_sections_hiddens, dim=2)
            # X_article.shape == [batch_size, src_len, 13, embed_size]

            weight = F.softmax(self.bert_weight_2, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            # weight.shape == [1, 1, 13, 1]

            X_article = (X_article * weight).sum(dim=2)
            # X_article.shape == [batch_size, seq_len, embed_size]

        else:
            X_article = X_sections_hiddens[-1]
            # X_article.shape == [batch_size, seq_len, embed_size]

        X_article = X_article.transpose(0, 1)
        # X_article.shape == [seq_len, batch_size, embed_size]

        Xs = []

        copied_indices_full = caption_ids.new_full((B, 1), -1)

        for i in range(gen_len):
            if i == 0:
                prev_target = {self.index: seed_input}
            else:
                prev_target = {self.index: seed_input[:, -1:]}

            self.decoder.filter_incremental_state(
                incremental_state, active_idx)

            contexts_i = {
                'image': contexts['image'][:, full_active_idx],
                'image_mask': contexts['image_mask'][full_active_idx],
                'article': contexts['article'][:, full_active_idx],
                'article_mask': contexts['article_mask'][full_active_idx],
                'faces': contexts['faces'][:, full_active_idx],
                'faces_mask': contexts['faces_mask'][full_active_idx],
                'sections':  None,
                'sections_mask': None,
            }

            decoder_out = self.decoder(
                prev_target,
                contexts_i,
                incremental_state=incremental_state)

            # We're only interested in the current final word
            decoder_out = (decoder_out[0][:, -1:], None)

            lprobs = self.decoder.get_normalized_probs(
                decoder_out, log_probs=True)
            # lprobs.shape == [batch_size, 1, vocab_size]

            lprobs = lprobs.squeeze(1)
            # lprobs.shape == [batch_size, vocab_size]

            X = decoder_out[0]
            # X.shape == [batch_size, 1, embed_size]

            Xs.append(X)

            X_full = torch.cat(Xs, dim=1)
            # X.shape == [batch_size_i, target_len, embed_size]

            X_full = X_full.transpose(0, 1)
            # X.shape == [target_len, batch_size_i, embed_size]

            X_entity = self.entity_attn(X_full)
            # X_entity.shape == [target_len, batch_size_i, embed_size]

            X_entity = X_entity.transpose(0, 1)
            # X_entity.shape == [batch_size_i, target_len, embed_size]

            entity_logits = self.entity_fc(X)
            # entity_logits.shape == [batch_size, 1, 2]

            entity_logits = self.entity_fc(X_entity[:, -1])
            # entity_logits.shape == [batch_size, 2]

            should_copy = entity_logits.argmax(dim=-1) == 1
            # should_copy.shape == [batch_size]

            X = X.transpose(0, 1)
            # X.shape == [1, batch_size, embed_size]

            X_article_i = X_article[:, full_active_idx]
            article_padding_mask_i = article_padding_mask[full_active_idx]

            copy_attn = multi_head_attention_score_forward(
                X, X_article_i, 1024, 16,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, True, 0.1, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=article_padding_mask_i)
            # copy_attn.shape == [batch_size, target_len, source_len + 2]

            copy_attn = copy_attn[:, :, :-2]
            # copy_attn.shape == [batch_size, 1, source_len]

            copy_attn = copy_attn.squeeze(1)
            # copy_attn.shape == [batch_size, source_len]

            irrelevant_mask = context_copy_masks[full_active_idx] != 1
            copy_attn[irrelevant_mask] = 0
            # copy_attn.shape == [batch_size, source_len]

            copy_probs = copy_attn.new_zeros(
                copy_attn.shape[0], self.vocab_size)
            # copy_probs.shape == [batch_size, vocab_size]

            context_ids = context[self.index][full_active_idx]
            # context_ids.shape == [batch_size, source_len]

            # First construct the reduced dictionary, containing only tokens
            # mentioned in the context.
            unique_ids = context_ids.unique()
            V = len(unique_ids)
            # unique_ids.shape == [reduced_vocab_size]

            # Construct the inverse map of unique_ids
            inverse_unique_ids = unique_ids.new_full([self.vocab_size], -1)
            inverse_unique_ids.index_copy_(
                0, unique_ids, torch.arange(V).to(unique_ids.device))
            # inverse_unique_ids.shape == [vocab_size]
            # e.g. [-1, -1, 0, -1, -1, 1, 2, -1, 3, ....]

            # Next we need to remap the context_ids to the new dictionary.
            new_context_ids = inverse_unique_ids.index_select(
                0, context_ids.reshape(-1))
            # new_context_ids.shape == [batch_size * source_len]

            B_i, S = copy_attn.shape
            new_context_ids = new_context_ids.view(B_i, S)
            # new_context_ids.shape == [batch_size, source_len]

            copy_probs = copy_attn.new_zeros(B_i, V)
            # copy_probs.shape == [batch_size, reduced_vocab_size]

            copy_probs.scatter_add_(1, new_context_ids, copy_attn)

            topk_copy_probs, topk_copy_indices = copy_probs.topk(
                self.sampling_topk)
            # topk_copy_probs.shape == [batch_size, topk]

            # If the top probability is 0, then we simply don't copy
            empty_copy = topk_copy_probs < 1e-6
            # Add small epsilon
            topk_copy_probs[empty_copy] = 1e-6
            should_copy = should_copy & (~empty_copy.max(dim=1)[0])

            sampled_copy_index = torch.multinomial(
                topk_copy_probs, num_samples=1)
            # sampled_copy_index.shape == [batch_size, 1]

            selected_copy_prob = topk_copy_probs.gather(
                dim=-1, index=sampled_copy_index)
            # selected_prob.shape == [batch_size, 1]

            selected_copy_new_index = topk_copy_indices.gather(
                dim=-1, index=sampled_copy_index)
            # selected_copy_new_index.shape == [batch_size, 1]

            # Convert back to old vocab space
            selected_copy_index = unique_ids.gather(
                dim=0, index=selected_copy_new_index.squeeze(1))
            # selected_copy_index.shape == [batch_size]

            selected_copy_index = selected_copy_index.unsqueeze(1)
            # selected_copy_index.shape == [batch_size, 1]

            selection = selected_copy_index.expand_as(
                copied_indices_full[full_active_idx])
            # selected_copy_index.shape == [batch_size_i, 1]

            has_copied = (
                selection == copied_indices_full[full_active_idx]).any(dim=1)
            # has_copied.shape == [batch_size_i]

            should_copy = should_copy & (~has_copied)
            # should_copy.shape == [batch_size_i]

            copied_indices = selected_copy_index.new_full((B, 1), -1)
            copied_indices[full_active_idx.nonzero().squeeze(1)[should_copy]] = \
                selected_copy_index[should_copy]
            copied_indices_full = torch.cat(
                [copied_indices_full, copied_indices], dim=1)

            copy_prob = selected_copy_prob.new_zeros(B, 1)
            copy_prob[full_active_idx] = selected_copy_prob

            copy_prob_list.append(copy_prob)

            should_copy_full = should_copy.new_zeros(B, 1)
            should_copy_full[full_active_idx] = should_copy.unsqueeze(1)
            should_copy_list.append(should_copy_full)

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

            selected_gen_index = topk_indices.gather(
                dim=-1, index=sampled_index)
            # selected_gen_index.shape == [batch_size, 1]

            selected_index = selected_gen_index.new_zeros(
                selected_gen_index.shape)
            selected_index[should_copy] = selected_copy_index[should_copy]
            selected_index[~should_copy] = selected_gen_index[~should_copy]

            log_prob = selected_lprob.new_zeros(B, 1)
            log_prob[full_active_idx] = selected_lprob

            index_path = selected_index.new_full((B, 1), self.padding_idx)
            index_path[full_active_idx] = selected_index

            log_prob_list.append(log_prob)
            index_path_list.append(index_path)

            seed_input = torch.cat([seed_input, selected_index], dim=-1)

            is_eos = selected_index.squeeze(-1) == eos
            active_idx = ~is_eos

            full_active_idx[full_active_idx.nonzero()[~active_idx]] = 0

            Xs = [x[active_idx] for x in Xs]

            seed_input = seed_input[active_idx]

            if active_idx.sum().item() == 0:
                break

        log_probs = torch.cat(log_prob_list, dim=-1)
        # log_probs.shape == [batch_size * beam_size, generate_len]

        copy_probs = torch.cat(copy_prob_list, dim=-1)

        should_copy_probs = torch.cat(should_copy_list, dim=-1)

        token_ids = torch.cat(index_path_list, dim=-1)
        # token_ids.shape == [batch_size * beam_size, generate_len]

        return log_probs, copy_probs, should_copy_probs, token_ids

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

        for key, value in self.batch_history.items():
            metrics[key] = value / self.n_batches

        if reset:
            self.n_batches = 0
            self.n_samples = 0
            self.sample_history: Dict[str, float] = defaultdict(float)
            self.batch_history: Dict[str, float] = defaultdict(float)

        return metrics
