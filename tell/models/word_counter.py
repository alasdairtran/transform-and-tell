from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from overrides import overrides


@Model.register("word_counter")
class WordCounter(Model):
    def __init__(self, vocab: Vocabulary, padding_value) -> None:
        super().__init__(vocab,)
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.padding_idx = padding_value
        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)
        self.placeholder = nn.Parameter(torch.Tensor(25))

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        article_ids = context['roberta']
        # article_ids.shape == [batch_size, seq_len]

        for article_id in article_ids:
            ids = article_id[article_id != self.padding_idx]
            text = self.roberta.decode(ids.cpu())
            n_words = text.split()
            self.sample_history['n_words'] += len(n_words)
            self.sample_history['n_tokens'] += ids.shape[0]

        output_dict = {
            'loss': None,
        }

        self.n_samples += article_ids.shape[0]
        self.n_batches += 1

        return output_dict

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
