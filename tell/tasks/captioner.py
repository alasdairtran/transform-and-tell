import base64
import io
import logging
import re
from collections import OrderedDict

import numpy as np
import spacy
import torch
from allennlp.common.util import prepare_environment
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from overrides import overrides
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

from tell.commands.train import yaml_to_params
from tell.data.fields import CopyTextField, ImageField
from tell.facenet import MTCNN, InceptionResnetV1

from .base import Worker

logger = logging.getLogger(__name__)

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class CaptioningWorker(Worker):
    def __init__(self, worker_id, worker_address_list, sink_address, verbose=False):
        super().__init__(worker_id, worker_address_list, sink_address, verbose)
        self.model = None
        self.bpe = None
        self.indices = None
        self.preprocess = None
        self.data_iterator = None
        self.tokenizer = None
        self.token_indexers = None
        self.mtcnn = None
        self.resnet = None
        self.nlp = None

    def initialize(self):
        # We need to initialize the model inside self.run and not self.__init__
        # to ensure that the model loads in the correct thread.
        config_path = '/home/alasdair/projects/transform-and-tell/expt/nytimes/7_transformer_faces/config.yaml'
        logger.info(f'Loading config from {config_path}')
        config = yaml_to_params(config_path, overrides='')
        prepare_environment(config)
        vocab = Vocabulary.from_params(config.pop('vocabulary'))
        model = Model.from_params(vocab=vocab, params=config.pop('model'))

        model_path = '/home/alasdair/projects/transform-and-tell/expt/nytimes/7_transformer_faces/serialization/best.th'
        logger.info(f'Loading best model from {model_path}')
        best_model_state = torch.load(model_path)
        model.load_state_dict(best_model_state)

        self.device = 0
        model.eval().to(self.device)
        self.model = model

        logger.info('Loading roberta model.')
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

        logger.info('Loading face detection model.')
        self.mtcnn = MTCNN(keep_all=True, device='cuda')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        self.preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        data_iterator = BasicIterator(batch_size=4)
        data_iterator.index_with(model.vocab)
        self.data_iterator = data_iterator

        self.tokenizer = Tokenizer.from_params(
            config.get('dataset_reader').get('tokenizer'))

        indexer_params = config.get('dataset_reader').get('token_indexers')

        self.token_indexers = {k: TokenIndexer.from_params(p)
                               for k, p in indexer_params.items()}

        # logger.info('Loading spacy')
        # self.nlp = spacy.load("en_core_web_lg")

    def generate_captions(self, articles):
        instances = [self.prepare_instance(a) for a in articles]
        iterator = self.data_iterator(instances, num_epochs=1, shuffle=False)
        generated_captions = []
        for batch in iterator:
            batch = move_to_device(batch, self.device)
            with torch.no_grad():
                output_dict = self.model.generate(**batch)
            generated_captions += output_dict['generations']

        output = []
        for i, instance in enumerate(instances):
            output.append({
                'title': instance['metadata']['title'],
                'start': instance['metadata']['start'],
                'before': instance['metadata']['before'],
                'after': instance['metadata']['after'],
                'caption': generated_captions[i],
            })

        return output

    def prepare_instance(self, article):
        sample = self.prepare_sample(article)

        context = '\n'.join(sample['paragraphs']).strip()

        context_tokens = self.tokenizer.tokenize(context)

        # proper_infos = self._get_context_names(context)

        fields = {
            # 'context': CopyTextField(context_tokens, self.token_indexers, proper_infos, proper_infos, 'context'),
            'context': TextField(context_tokens, self.token_indexers),
            'image': ImageField(sample['image'], self.preprocess),
            'face_embeds': ArrayField(sample['face_embeds'], padding_value=np.nan),
        }

        metadata = {
            'title': sample['title'],
            'start': '\n'.join(sample['start']).strip(),
            'before': '\n'.join(sample['before']).strip(),
            'after': '\n'.join(sample['after']).strip(),
        }
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    # def _get_context_names(self, text):
    #     copy_infos = {}

    #     doc = self.nlp(text)
    #     parts_of_speech = []
    #     for tok in doc:
    #         pos = {
    #             'start': tok.idx,
    #             'end': tok.idx + len(tok.text),  # exclude right endpoint
    #             'text': tok.text,
    #             'pos': tok.pos_,
    #         }
    #         parts_of_speech.append(pos)

    #     for pos in parts_of_speech:
    #         if pos['pos'] == 'PROPN':
    #             if pos['text'] not in copy_infos:
    #                 copy_infos[pos['text']] = OrderedDict({
    #                     'context': [(pos['start'], pos['end'])]
    #                 })
    #             else:
    #                 copy_infos[pos['text']]['context'].append(
    #                     (pos['start'], pos['end']))

    #     return copy_infos

    def prepare_sample(self, article):
        paragraphs = []
        start = []
        n_words = 0
        pos = article['image_position']
        sections = article['sections']

        if article['title']:
            paragraphs.append(article['title'])
            n_words += len(self.to_token_ids(article['title']))

        before = []
        after = []
        i = pos - 1
        j = pos + 1

        # Append the first paragraph
        for k, section in enumerate(sections):
            if section['type'] == 'paragraph':
                paragraphs.append(section['text'])
                start.append(section['text'])
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

        image_data = base64.b64decode(
            sections[pos]['image_data'].encode('utf-8'))
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        face_embeds = self.get_faces(image)

        output = {
            'paragraphs': paragraphs + before + after,
            'title': article['title'],
            'start': start,
            'before': before,
            'after': after,
            'image': image,
            'face_embeds': face_embeds,
        }

        return output

    def get_faces(self, image):
        with torch.no_grad():
            try:
                faces = self.mtcnn(image)
            except IndexError:  # Strange index error on line 135 in utils/detect_face.py
                logger.warning('Strange index error from FaceNet.')
                return np.array([[]])

            if faces is None:
                return np.array([[]])

            embeddings, _ = self.resnet(faces)
            return embeddings.cpu().numpy()[:4]

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids

    @overrides
    def _process(self, job):
        articles = job['message']
        output = self.generate_captions(articles)

        return {
            'client_id': job['client_id'],
            'output': output,
        }
