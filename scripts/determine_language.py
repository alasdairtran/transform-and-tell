"""Get articles from the New York Times API.

Usage:
    determine_language.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from langdetect import detect
from newser.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
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

    client = MongoClient(host='localhost', port=27017)

    article_cursor = client.nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Determine article languages in NYTimes dataset.')
    for article in tqdm(article_cursor):
        if 'language' in article:
            continue

        text_list = [sec['text'] for sec in article['parsed_section']]
        text = '\n'.join(text_list)
        article['language'] = detect(text)

        client.nytimes.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})

    article_cursor = client.goodnews.articles.find(
        {}, no_cursor_timeout=True).batch_size(128)

    logger.info('Determine article languages in GoodNews dataset.')
    for article in tqdm(article_cursor):
        if 'language' in article or 'article' not in article:
            continue

        article['language'] = detect(article['article'])

        client.goodnews.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


if __name__ == '__main__':
    main()
