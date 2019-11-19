import logging
import os
import random
import re
from datetime import datetime
from typing import Dict

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

from tell.data.fields import ImageField, ListTextField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@DatasetReader.register('nytimes_summary')
class NYTimesSummaryReader(DatasetReader):
    """Read from the New York Times dataset.

    See the repo README for more instruction on how to download the dataset.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.nytimes
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        random.seed(1234)

        roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split == 'train':
            start = datetime(2000, 1, 1)
            end = datetime(2019, 5, 1)
        elif split == 'valid':
            start = datetime(2019, 5, 1)
            end = datetime(2019, 6, 1)
        elif split == 'test':
            start = datetime(2019, 6, 1)
            end = datetime(2019, 9, 1)
        else:
            raise ValueError(f'Unknown split: {split}')

        # Setting the batch size is needed to avoid cursor timing out
        article_cursor = self.db.articles.find({
            'parsed': True,  # article body is parsed into paragraphs
            'n_images': {'$gt': 0},  # at least one image is present
            'pub_date': {'$gte': start, '$lt': end},
            'language': 'en',
        }, no_cursor_timeout=True).batch_size(128)

        for article in article_cursor:
            sections = article['parsed_section']
            image_positions = article['image_positions']
            for pos in image_positions:
                title = ''
                if 'main' in article['headline']:
                    title = article['headline']['main'].strip()
                paragraphs = []
                n_words = 0
                if title:
                    paragraphs.append(title)
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                before = []
                after = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        n_words += len(self.to_token_ids(text))
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
                        n_words += len(self.to_token_ids(text))
                    j += 1

                    if n_words >= 510 or (i <= k and j >= len(sections)):
                        break

                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                paragraphs = paragraphs + before + after

                all_paragraphs = [s['text'].strip()
                                  for s in sections if s['type'] == 'paragraph']
                if title:
                    all_paragraphs.insert(0, title)

                yield self.article_to_instance(paragraphs, all_paragraphs, image, caption, image_path, article['web_url'], pos)

        article_cursor.close()

    def article_to_instance(self, paragraphs, all_paragraphs, image, caption, image_path, web_url, pos) -> Instance:
        context = '\n'.join(paragraphs).strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        all_context_tokens_list = [
            self._tokenizer.tokenize(par) for par in all_paragraphs]

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            'full_context': ListTextField([TextField(par, self._token_indexers)
                                           for par in all_context_tokens_list]),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
        }

        metadata = {'context': context,
                    'full_context':  '\n'.join(all_paragraphs).strip(),
                    'caption': caption,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids
