"""Get articles from the New York Times API.

Usage:
    clean_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]

"""
import os
from datetime import datetime

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

    article_cursor = nytimes.text_articles.find({
    }, no_cursor_timeout=True).batch_size(128)
    for article in tqdm(article_cursor):
        if 'raw_html' not in article:
            continue
        source = {
            '_id': article['_id'],
            'raw_html': article['raw_html']
        }
        nytimes.source.find_one_and_update(
            {'_id': article['_id']}, {'$set': source})
        del article['raw_html']
        nytimes.text_articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})

    article_cursor = nytimes.articles.find({
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Processing NYTimes+')
    for article in tqdm(article_cursor):
        if 'facenet_positions' in article:
            del article['facenet_positions']
        if 'n_images_with_dfsd_faces' in article:
            del article['n_images_with_dfsd_faces']
        if 'face_positions' in article:
            del article['face_positions']

        if 'parsed_section' in article:
            for s in article['parsed_section']:
                if 'facenet' in s:
                    del s['facenet']
                if 'faces' in s:
                    del s['faces']
                if 'n_faces' in s:
                    del s['n_faces']

        # Save source separately
        if 'raw_html' in article:
            source = {
                '_id': article['_id'],
                'raw_html': article['raw_html']
            }
            nytimes.source.find_one_and_update(
                {'_id': article['_id']}, {'$set': source})
            del article['raw_html']

        if not article['parsed'] or article['n_images'] == 0 or article['language'] != 'en':
            nytimes.text_articles.insert_one(article)
            nytimes.articles.delete_one({'_id': article['_id']})
        else:
            nytimes.articles.find_one_and_update(
                {'_id': article['_id']}, {'$set': article})

    logger.info('Building NYTimes+ indices')
    nytimes.articles.create_index([
        ('n_images', pymongo.ASCENDING),
        ('n_images_with_faces', pymongo.ASCENDING),
        ('pub_date', pymongo.DESCENDING),
    ])

    nytimes.articles.create_index([
        ('pub_date', pymongo.DESCENDING),
    ])

    sample_cursor = client.goodnews.splits.find({
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Processing Good News')
    for sample in tqdm(sample_cursor):
        if 'facenet' in sample:
            del sample['facenet']
            client.goodnews.splits.find_one_and_update(
                {'_id': sample['_id']}, {'$set': sample})


if __name__ == '__main__':
    main()
