"""Get articles from the New York Times API.

Usage:
    clean_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]

"""
import os
from datetime import datetime

import numpy as np
import ptvsd
import pymongo
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
        'host': str,
    })
    args = schema.validate(args)
    return args


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    client = MongoClient(host=args['host'], port=27017)
    nytimes = client.nytimes

    article_cursor = nytimes.articles.find({
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        sections = article['parsed_section']
        image_positions = article['image_positions']
        for pos in image_positions:
            s = sections[pos]
            if 'facenet_details' in s:
                face_probs = np.array(s['facenet_details']['face_probs'])
                face_prob_list = []
                for face_prob in face_probs:
                    top10_idx = np.argpartition(face_prob, -10)[-10:]
                    top10 = face_prob[top10_idx]
                    top10_idx_sorted = top10_idx[np.argsort(top10)[
                        ::-1]].tolist()
                    top10_sorted = face_prob[top10_idx_sorted].tolist()
                    fpl = [[i, p]
                           for i, p in zip(top10_idx_sorted, top10_sorted)]
                    face_prob_list.append(fpl)

                s['facenet_details']['face_probs'] = face_prob_list

        nytimes.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


if __name__ == '__main__':
    main()
