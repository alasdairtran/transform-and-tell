import logging
import os
import random
from typing import Dict

import numpy as np
import pymongo
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from newser.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('goodnews_flattened_glove')
class FlattenedGloveGoodNewsReader(DatasetReader):
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
                 annotation_path: str = None,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 eval_limit: int = 5120,
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
        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        # Setting the batch size is needed to avoid cursor timing out
        # We limit the validation set to 1000
        logger.info('Grabbing all article IDs')
        limit = self.eval_limit if split == 'val' else 0
        sample_cursor = self.db.splits.find({
            'split': {'$eq': split},
        }, projection=['_id'], limit=limit).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        self.rs.shuffle(ids)

        for sample_id in ids:
            sample = self.db.splits.find_one({'_id': {'$eq': sample_id}})

            # Find the corresponding article
            article = self.db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            }, projection=['_id', 'context', 'images', 'web_url'])

            # Load the image
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            yield self.article_to_instance(article, image, sample['image_index'], image_path)

    def article_to_instance(self, article, image, image_index, image_path) -> Instance:
        context = article['context'].strip()
        caption = article['images'][image_index]
        caption = caption.strip()

        caption_tokens = self._tokenizer.tokenize(caption)

        fields = {
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
        }

        metadata = {'context': context,
                    'caption': caption,
                    'web_url': article['web_url'],
                    'image_path': image_path}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
