"""Get articles from the New York Times API.

Usage:
    detect_facenet.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -d --image-dir DIR  Image directory [default: ./data/nytimes/images].
    -f --face-dir DIR   Image directory [default: ./data/nytimes/facenet].
    -b --batch INT      Batch number [default: 1]
    -h --host HOST      Mongo host name [default: localhost]

"""
import os
from datetime import datetime

import numpy as np
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
    if 'detected_face_positions' in article:
        return

    sections = article['parsed_section']
    image_positions = article['image_positions']
    article['detected_face_positions'] = []

    for pos in image_positions:
        section = sections[pos]
        image_path = os.path.join(image_dir, f"{section['hash']}.jpg")
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path} from article "
                           f"{article['_id']} at position {pos}")
            continue

        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except OSError:
            logger.warning(f"OSError on image: {image_path} from article "
                           f"{article['_id']} at position {pos}")
            continue
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

        face_probs = face_probs.cpu().numpy()
        face_prob_list = []
        for face_prob in face_probs:
            top10_idx = np.argpartition(face_prob, -10)[-10:]
            top10 = face_prob[top10_idx]
            top10_idx_sorted = top10_idx[np.argsort(top10)[::-1]].tolist()
            top10_sorted = face_prob[top10_idx_sorted].tolist()
            fpl = [(i, p) for i, p in zip(top10_idx_sorted, top10_sorted)]
            face_prob_list.append(fpl)

        section['facenet_details'] = {
            'n_faces': len(faces[:10]),
            'embeddings': embeddings.cpu().tolist()[:10],
            'detect_probs': probs.tolist()[:10],
            'face_probs': face_prob_list,
        }

        article['detected_face_positions'].append(pos)

    article['n_images_with_faces'] = len(article['detected_face_positions'])

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

    # if args['batch'] == 1:  # 38K
    #     start = datetime(2000, 1, 1)
    #     end = datetime(2007, 8, 1)
    # elif args['batch'] == 2:  # 42K
    #     start = datetime(2007, 8, 1)
    #     end = datetime(2009, 1, 1)
    # elif args['batch'] == 3:  # 41K
    #     start = datetime(2009, 1, 1)
    #     end = datetime(2010, 5, 1)
    # elif args['batch'] == 4:  # 79K
    #     start = datetime(2010, 5, 1)
    #     end = datetime(2012, 11, 1)
    # elif args['batch'] == 5:  # 80K
    #     start = datetime(2012, 11, 1)
    #     end = datetime(2015, 5, 1)
    # elif args['batch'] == 6:  # 80K
    #     start = datetime(2015, 5, 1)
    #     end = datetime(2017, 5, 1)
    # elif args['batch'] == 7:  # 81K
    #     start = datetime(2017, 5, 1)
    #     end = datetime(2019, 9, 1)
    if args['batch'] == 1:
        start = datetime(2000, 1, 1)
        end = datetime(2013, 2, 1)
    elif args['batch'] == 2:
        start = datetime(2013, 2, 1)
        end = datetime(2019, 9, 1)
    else:
        raise ValueError(f"Unknown batch: {args['batch']}")

    article_cursor = nytimes.articles.find({
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
