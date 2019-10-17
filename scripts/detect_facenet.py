"""Get articles from the New York Times API.

Usage:
    detect_facenet.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -d --image-dir DIR  Image directory [default: ./data/nytimes/images_processed].
    -f --face-dir DIR   Image directory [default: ./data/nytimes/facenet].
    -b --batch INT      Batch number [default: 1]
    -h --host HOST      Mongo host name [default: localhost]

"""
import os
from datetime import datetime

import ptvsd
import torch
from docopt import docopt
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import DocumentTooLarge
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.facenet import MTCNN, InceptionResnetV1
from newser.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'image_dir': str,
        'face_dir': str,
        'batch': Use(int),
        'host': str,
    })
    args = schema.validate(args)
    return args


def detect_faces(article, nytimes, image_dir, face_dir, mtcnn, resnet):
    if 'facenet_positions' in article:
        return

    sections = article['parsed_section']
    image_positions = article['image_positions']
    article['facenet_positions'] = []

    for pos in image_positions:
        section = sections[pos]
        image_path = os.path.join(image_dir, f"{section['hash']}.jpg")
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path} from article "
                           f"{article['_id']} at position {pos}")
            continue

        img = Image.open(image_path)
        face_path = os.path.join(face_dir, f"{section['hash']}_{pos:02}.jpg")
        with torch.no_grad():
            try:
                faces, probs = mtcnn(img, save_path=face_path,
                                     return_prob=True)
            except IndexError:  # Strange index error on line 135 in utils/detect_face.py
                logger.warning(f"IndexError on image: {image_path} from article "
                               f"{article['_id']} at position {pos}")
                faces = None
            if faces is None:
                continue
            embeddings, face_probs = resnet(faces)

        section['facenet'] = {
            'n_faces': len(faces),
            'embeddings': embeddings.cpu().tolist(),
            'detect_probs': probs.tolist(),
            'face_probs': face_probs.cpu().tolist(),
        }

        article['facenet_positions'].append(pos)

    article['n_images_with_faces'] = len(article['facenet_positions'])

    try:
        nytimes.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})
    except DocumentTooLarge:
        logger.warning(f"Document too large: {article['_id']}")


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)
    image_dir = args['image_dir']
    face_dir = args['face_dir']

    os.makedirs(face_dir, exist_ok=True)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    client = MongoClient(host=args['host'], port=27017)
    nytimes = client.nytimes

    if args['batch'] == 1:  # 38K
        start = datetime(2000, 1, 1)
        end = datetime(2007, 8, 1)
    elif args['batch'] == 2:  # 42K
        start = datetime(2007, 8, 1)
        end = datetime(2009, 1, 1)
    elif args['batch'] == 3:  # 41K
        start = datetime(2009, 1, 1)
        end = datetime(2010, 5, 1)
    elif args['batch'] == 4:  # 79K
        start = datetime(2010, 5, 1)
        end = datetime(2012, 11, 1)
    elif args['batch'] == 5:  # 80K
        start = datetime(2012, 11, 1)
        end = datetime(2015, 5, 1)
    elif args['batch'] == 6:  # 80K
        start = datetime(2015, 5, 1)
        end = datetime(2017, 5, 1)
    elif args['batch'] == 7:  # 81K
        start = datetime(2017, 5, 1)
        end = datetime(2019, 9, 1)
    else:
        raise ValueError(f"Unknown batch: {args['batch']}")

    article_cursor = nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'language': 'en',
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Loading model.')
    mtcnn = MTCNN(keep_all=True, device='cuda')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    logger.info('Detecting faces.')
    for article in tqdm(article_cursor):
        detect_faces(article, nytimes, image_dir, face_dir, mtcnn, resnet)

    # with Parallel(n_jobs=8, backend='threading') as parallel:
    #     parallel(delayed(detect_faces)(article, nytimes, image_dir, face_dir, mtcnn, resnet)
    #              for article in tqdm(article_cursor))


if __name__ == '__main__':
    main()
