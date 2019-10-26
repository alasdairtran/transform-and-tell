"""Get articles from the New York Times API.

Usage:
    clean_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]

"""
import functools
import os
from datetime import datetime
from multiprocessing import Pool

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


def clean_with_host(host, period):
    start, end = period
    client = MongoClient(host=host, port=27017)
    db = client.nytimes

    article_cursor = db.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        sections = article['parsed_section']
        image_positions = article['image_positions']
        for pos in image_positions:
            s = sections[pos]
            if 'facenet_details' in s and 'face_probs' in s['facenet_details']:
                del s['facenet_details']['face_probs']

        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    periods = [(datetime(2000, 1, 1), datetime(2007, 8, 1)),
               (datetime(2007, 8, 1), datetime(2009, 1, 1)),
               (datetime(2009, 1, 1), datetime(2010, 5, 1)),
               (datetime(2010, 5, 1), datetime(2011, 7, 1)),
               (datetime(2011, 7, 1), datetime(2012, 11, 1)),
               (datetime(2012, 11, 1), datetime(2014, 2, 1)),
               (datetime(2014, 2, 1), datetime(2015, 5, 1)),
               (datetime(2015, 5, 1), datetime(2016, 5, 1)),
               (datetime(2016, 5, 1), datetime(2017, 5, 1)),
               (datetime(2017, 5, 1), datetime(2018, 8, 1)),
               (datetime(2018, 8, 1), datetime(2019, 9, 1)),
               ]

    clean = functools.partial(clean_with_host, args['host'])

    pool = Pool(processes=11)
    pool.map(clean, periods)


if __name__ == '__main__':
    main()
