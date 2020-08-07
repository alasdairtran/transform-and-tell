import base64
import io
import logging
import os
import random
import re
from collections import OrderedDict
from io import BytesIO

import cv2
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
from tell.data.fields import ImageField
from tell.facenet import MTCNN, InceptionResnetV1
from tell.models.resnet import resnet152
from tell.yolov3.models import Darknet, attempt_download
from tell.yolov3.utils.datasets import letterbox
from tell.yolov3.utils.utils import (load_classes, non_max_suppression,
                                     plot_one_box, scale_coords)

from .base import Worker

logger = logging.getLogger(__name__)
SPACE_NORMALIZER = re.compile(r"\s+")
ENV = os.environ.copy()


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
        self.inception = None
        self.resnet = None
        self.darknet = None
        self.names = None
        self.colors = None
        self.nlp = None
        if torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
            d = worker_id % n_devices
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                devs = ENV['CUDA_VISIBLE_DEVICES'].split(',')
                os.environ['CUDA_VISIBLE_DEVICES'] = devs[d]
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(d)
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device('cpu')

    def initialize(self):
        # We need to initialize the model inside self.run and not self.__init__
        # to ensure that the model loads in the correct thread.
        config_path = 'expt/nytimes/9_transformer_objects/config.yaml'
        logger.info(f'Loading config from {config_path}')
        config = yaml_to_params(config_path, overrides='')
        prepare_environment(config)
        vocab = Vocabulary.from_params(config.pop('vocabulary'))
        model = Model.from_params(vocab=vocab, params=config.pop('model'))
        model = model.eval()

        model_path = 'expt/nytimes/9_transformer_objects/serialization/best.th'
        logger.info(f'Loading best model from {model_path}')
        best_model_state = torch.load(
            model_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_model_state)

        self.model = model.to(self.device)

        logger.info('Loading roberta model.')
        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

        logger.info('Loading face detection model.')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.inception = InceptionResnetV1(pretrained='vggface2').eval()

        self.resnet = resnet152()
        self.resnet = self.resnet.to(self.device).eval()

        cfg = 'tell/yolov3/cfg/yolov3-spp.cfg'
        weight_path = 'data/yolov3-spp-ultralytics.pt'
        self.darknet = Darknet(cfg, img_size=416)
        attempt_download(weight_path)
        self.darknet.load_state_dict(torch.load(
            weight_path, map_location=self.device)['model'])
        self.darknet.to(self.device).eval()

        # Get names and colors
        self.names = load_classes('tell/yolov3/data/coco.names')
        random.seed(123)
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]

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
            if self.device.type == 'cuda':
                batch = move_to_device(batch, self.device.index)
            attns_list = self.model.generate(**batch)
            # generated_captions += output_dict['generations']
            # attns = output_dict['attns']
            # len(attns) == gen_len (ignoring seed)
            # len(attns[0]) == n_layers
            # attns[0][0]['image'].shape == [47]
            # attns[0][0]['article'].shape == [article_len]

        output = []
        for i, instance in enumerate(instances):
            buffered = BytesIO()
            instance['metadata']['image'].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            output.append({
                'title': instance['metadata']['title'],
                'start': instance['metadata']['start'],
                'before': instance['metadata']['before'],
                'after': instance['metadata']['after'],
                # 'caption': generated_captions[i],
                'attns': attns_list[i],
                'image': img_str,
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
            'obj_embeds': ArrayField(sample['obj_embeds'], padding_value=np.nan),
        }

        metadata = {
            'title': sample['title'],
            'start': '\n'.join(sample['start']).strip(),
            'before': '\n'.join(sample['before']).strip(),
            'after': '\n'.join(sample['after']).strip(),
            'image': CenterCrop(224)(Resize(256)(sample['image']))
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
        obj_embeds = self.get_objects(image)

        output = {
            'paragraphs': paragraphs + before + after,
            'title': article['title'],
            'start': start,
            'before': before,
            'after': after,
            'image': image,
            'face_embeds': face_embeds,
            'obj_embeds': obj_embeds,
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

            embeddings, _ = self.inception(faces)
            return embeddings.cpu().numpy()[:4]

    def get_objects(self, image):
        im0 = np.array(image)
        img = letterbox(im0, new_shape=416)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.darknet(img)[0]

        # Apply NMS
        # We ignore the person class (class 0)
        pred = non_max_suppression(pred, 0.3, 0.6,
                                   classes=None, agnostic=False)

        # Process detections
        assert len(pred) == 1, f'Length of pred is {len(pred)}'
        det = pred[0]

        im0 = im0[:, :, ::-1]  # to BGR
        im0 = np.ascontiguousarray(im0)

        obj_feats = []
        confidences = []
        classes = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for j, (*xyxy, conf, class_) in enumerate(det):
                if j >= 64:
                    break

                obj_feat = get_obj_embeddings(
                    xyxy, image, None, self.resnet)
                obj_feats.append(obj_feat)
                confidences.append(conf.item())
                classes.append(int(class_))

                label = '%s %.2f' % (self.names[int(class_)], conf)
                plot_one_box(xyxy, im0, label=label,
                             color=self.colors[int(class_)])

        # Save results (image with detections)
        # cv2.imwrite(save_path, im0)

        if not obj_feats:
            return np.array([[]])
        return np.array(obj_feats)

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
        with torch.no_grad():
            output = self.generate_captions(articles)

        return {
            'client_id': job['client_id'],
            'output': output,
        }


def get_obj_embeddings(xyxy, pil_image, obj_path, resnet):
    pil_image = pil_image.convert('RGB')
    obj_image = extract_object(pil_image, xyxy, save_path=obj_path)

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    obj_image = preprocess(obj_image)
    # obj_image.shape == [n_channels, height, width]

    # Add a batch dimension
    obj_image = obj_image.unsqueeze(0).to(next(resnet.parameters()).device)
    # obj_image.shape == [1, n_channels, height, width]

    X_image = resnet(obj_image, pool=True)
    # X_image.shape == [1, 2048]

    X_image = X_image.squeeze(0).cpu().numpy().tolist()
    # X_image.shape == [2048]

    return X_image


def extract_object(img, box, image_size=224, margin=0, save_path=None):
    """Extract object + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted object image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted object.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin)
    ]
    box = [
        int(max(box[0] - margin[0]/2, 0)),
        int(max(box[1] - margin[1]/2, 0)),
        int(min(box[2] + margin[0]/2, img.size[0])),
        int(min(box[3] + margin[1]/2, img.size[1]))
    ]

    obj = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path)+'/', exist_ok=True)
        save_args = {'compress_level': 0} if '.png' in save_path else {}
        obj.save(save_path, **save_args)

    return obj
