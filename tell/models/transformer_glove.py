import math
import re
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import spacy
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

from tell.modules.criteria import Criterion

from .decoder_flattened import Decoder
from .resnet import resnet152


@Model.register("transformer_glove")
class TransformerGloveModel(Model):
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
        self.resnet = resnet152()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)

        initializer(self)

        self.nlp = spacy.load('en_core_web_lg',
                              disable=['textcat', 'parser', 'tagger', 'ner'])

        # Initialize the weight with first layer of BERT
        # self.fc.weight.data.copy_(
        #     self.roberta.model.decoder.sentence_encoder.embed_tokens.weight)

    def forward(self,  # type: ignore
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        context = [m['context'] for m in metadata]
        caption_ids, target_ids, contexts = self._forward(
            context, image, caption)
        decoder_out = self.decoder(caption, contexts)

        # Assume we're using adaptive loss
        loss, sample_size = self.criterion(
            self.decoder.adaptive_softmax, decoder_out, target_ids)

        loss = loss / math.log(2)

        output_dict = {
            'loss': loss / sample_size,
            'sample_size': sample_size,
        }

        # During evaluation, we will generate a caption and compute BLEU, etc.
        if not self.training and self.evaluate_mode:
            _, gen_ids = self._generate(caption_ids, contexts)
            # We ignore <s> and <pad>
            gen_texts = [self.roberta.decode(x[x > 1]) for x in gen_ids.cpu()]
            captions = [m['caption'] for m in metadata]

            output_dict['captions'] = captions
            output_dict['generations'] = gen_texts
            output_dict['metadata'] = metadata

            # Remove punctuation
            gen_texts_2 = [re.sub(r'[^\w\s]', '', t) for t in gen_texts]
            captions_2 = [re.sub(r'[^\w\s]', '', t) for t in captions]

            for gen, ref in zip(gen_texts_2, captions_2):
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

            if 'rare_tokens' in caption:
                for gen, ref, rare_list in zip(gen_texts_2, captions_2, caption['rare_tokens']):
                    bleu_scorer = BleuScorer(n=4)
                    rare_words = ' '.join(rare_list)
                    gen = gen + ' ' + rare_words

                    if rare_words:
                        print(ref)
                        print(gen)
                        print()

                    bleu_scorer += (gen, [ref])
                    score, _ = bleu_scorer.compute_score(option='closest')
                    self.sample_history['bleu-1r'] += score[0] * 100

        self.n_samples += caption_ids.shape[0]
        self.n_batches += 1

        return output_dict

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        caption_ids, _, contexts = self._forward(
            context, image, caption)

        _, gen_ids = self._generate(caption_ids, contexts)

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

        # article_ids.shape == [batch_size, seq_len]

        context = [c.lower() for c in context]
        context_docs = self.nlp.pipe(context)
        vs = []
        v_lens = []
        for doc in context_docs:
            v = [token.vector for token in doc if token.has_vector]
            v_lens.append(len(v))
            vs.append(np.array(v))
        max_len = max(v_lens)

        context_vector = X_image.new_full((B, max_len, 300), np.nan)
        for i, v in enumerate(vs):
            v_len = v.shape[0]
            v_tensor = torch.from_numpy(v).type_as(context_vector)
            context_vector[i, :v_len] = v_tensor

        article_padding_mask = torch.isnan(context_vector).any(dim=-1)
        # article_padding_mask.shape == [batch_size, seq_len]

        X_article = context_vector
        X_article[article_padding_mask] = 0
        # Create padding mask (1 corresponds to the padding index)
        image_padding_mask = X_image.new_zeros(B, P).bool()

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
        }

        return caption_ids, target_ids, contexts

    def _generate(self, caption_ids, contexts):
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

            contexts_i = {
                'image': contexts['image'][:, full_active_idx],
                'image_mask': contexts['image_mask'][full_active_idx],
                'article': contexts['article'][:, full_active_idx],
                'article_mask': contexts['article_mask'][full_active_idx],
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

            index_path = selected_index.new_full((B, 1), self.padding_idx)
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
