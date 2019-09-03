import logging
import os
import pickle
import random
from typing import Dict

import spacy
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from spacy.tokens import Doc
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from newser.data.fields import ImageField, ListTextField, SpacyTextField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('goodnews_flattened')
class FlattenedGoodNewsReader(DatasetReader):
    """Read from the Good News dataset.

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
                 annotation_path: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 eval_limit: int = 5120,
                 cache_annotations: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.goodnews
        self.image_dir = image_dir
        self.preprocess = Compose([
            # Resize(256), CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.eval_limit = eval_limit
        random.seed(1234)

        logger.info('Loading spacy model.')
        nlp = spacy.load("en_core_web_lg", disable=['parser', 'tagger'])
        self.spacy_vocab = nlp.vocab

        logger.info('Loading entity annotations.')
        ann_path = os.path.join(annotation_path)
        with open(ann_path, 'rb') as f:
            self.annotations = pickle.load(f)

        self.cache_annotations = cache_annotations
        if cache_annotations:
            logger.info('Converting annotations to Doc objects.')
            for ann in tqdm(self.annotations.values()):
                ann['context'] = Doc(nlp.vocab).from_bytes(ann['context'])
                ann['captions'] = {k: Doc(nlp.vocab).from_bytes(c)
                                   for k, c in ann['captions'].items()}

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        # Setting the batch size is needed to avoid cursor timing out
        # We limit the validation set to 1000
        limit = self.eval_limit if split == 'val' else 0
        sample_cursor = self.db.splits.find({
            'split': {'$eq': split},
        }, no_cursor_timeout=True, limit=limit).batch_size(128)

        for sample in sample_cursor:
            # Find the corresponding article
            article = self.db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            })

            # Load the image
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            ann = self.annotations[sample['article_id']]
            yield self.article_to_instance(article, image, sample['image_index'], ann)

        sample_cursor.close()

    def article_to_instance(self, article, image, image_index, ann) -> Instance:
        context = article['context'].strip()
        context_doc = ann['context']

        caption = article['images'][image_index]
        caption = caption.strip()
        caption_doc = ann['captions'][image_index]

        if not self.cache_annotations:
            context_doc = Doc(self.spacy_vocab).from_bytes(context_doc)
            caption_doc = Doc(self.spacy_vocab).from_bytes(caption_doc)

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        fields = {
            'context': SpacyTextField(context_tokens, self._token_indexers, context_doc),
            'image': ImageField(image, self.preprocess),
            'caption': SpacyTextField(caption_tokens, self._token_indexers, caption_doc),
        }

        metadata = {'context': context,
                    'caption': caption,
                    'web_url': article['web_url']}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
