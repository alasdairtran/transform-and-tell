"""Get articles from the New York Times API.

Usage:
    clean_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]

"""

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

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


def clean_with_host(host):
    client = MongoClient(host=host, port=27017)
    nytimes = client.nytimes

    # article_cursor = nytimes.articles.find({})
    # for article in tqdm(article_cursor):
    #     changed = False
    #     if 'raw_html' in article:
    #         source = {
    #             '_id': article['_id'],
    #             'raw_html': article['raw_html']
    #         }
    #         nytimes.source.find_one_and_update(
    #             {'_id': article['_id']}, {'$set': source}, upsert=True)

    #         del article['raw_html']
    #         changed = True

    #     if 'facenet_positions' in article:
    #         del article['facenet_positions']
    #         changed = True

    #     if 'n_images_with_dfsd_faces' in article:
    #         del article['n_images_with_dfsd_faces']
    #         changed = True

    #     if 'face_positions' in article:
    #         del article['face_positions']
    #         changed = True

    #     if 'coref_clusters' in article:
    #         del article['coref_clusters']
    #         changed = True

    #     for s in article['parsed_section']:
    #         if 'corefs' in s:
    #             del s['corefs']
    #             changed = True
    #         if 'faces' in s:
    #             del s['faces']
    #             changed = True
    #         if 'facenet' in s:
    #             del s['facenet']
    #             changed = True
    #         if 'n_faces' in s:
    #             del s['n_faces']
    #             changed = True
    #         if 'facenet_details' in s and 'face_probs' in s['facenet_details']:
    #             del s['facenet_details']['face_probs']
    #             changed = True

    #     if 'main' in article['headline'] and 'corefs' in article['headline']:
    #         del article['headline']['corefs']
    #         changed = True

    #     if changed:
    #         nytimes.articles.replace_one({'_id': article['_id']}, article)

    article_cursor = nytimes.text_articles.find({})
    for article in tqdm(article_cursor):
        if 'raw_html' not in article:
            continue
        source = {
            '_id': article['_id'],
            'raw_html': article['raw_html']
        }
        nytimes.source.find_one_and_update(
            {'_id': article['_id']}, {'$set': source}, upsert=True)
        del article['raw_html']
        nytimes.text_articles.replace_one(
            {'_id': article['_id']}, article)

    goodnews = client.goodnews
    sample_cursor = goodnews.splits.find({})
    for sample in tqdm(sample_cursor):
        if 'facenet' in sample:
            del sample['facenet']
            goodnews.splits.replace_one({'_id': sample['_id']}, sample)


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    clean_with_host(args['host'])


if __name__ == '__main__':
    main()
