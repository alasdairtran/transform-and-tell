"""Get articles from the New York Times API.

Usage:
    compute_data_statistics.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""
import os
from datetime import datetime
from glob import glob

import ptvsd
import torchvision.transforms.functional as F
from docopt import docopt
from PIL import Image
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


def compute_nytimes_stats(nytimes):
    cursor = nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'language': 'en',
    })

    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_articles = 0

    max_date = datetime(2000, 1, 1)
    min_date = datetime(2020, 1, 1)

    logger.info('Computing NYTimes statistics.')
    for article in tqdm(cursor):
        sections = article['parsed_section']
        pars = [s['text'] for s in sections if s['type'] == 'paragraph']
        captions = []

        for s in sections:
            if s['type'] == 'caption':
                image_path = os.path.join('data/nytimes/images_processed',
                                          f"{s['hash']}.jpg")
                if os.path.exists(image_path):
                    captions.append(s['text'])
        if not captions:
            continue

        n_article_words += len(' '.join(pars).split())
        n_articles += 1
        n_caption_words += len(' '.join(captions).split())
        n_captions += len(captions)

        if article['pub_date'] < min_date:
            min_date = article['pub_date']
        if article['pub_date'] > max_date:
            max_date = article['pub_date']

    print('Full NYTimes Dataset:')
    print('No of articles:', n_articles)
    print('No of captions:', n_captions)
    print('Average article len:', n_article_words / n_articles)
    print('Average caption len:', n_caption_words / n_captions)
    print('Min date:', min_date)
    print('Max date:', max_date)
    print()


def compute_nytimes_subset_statistics(nytimes):
    cursor = nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'pub_date': {'$gte': datetime(2010, 1, 1), '$lt': datetime(2018, 7, 1)}
    })

    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_articles = 0

    max_date = datetime(2000, 1, 1)
    min_date = datetime(2020, 1, 1)

    logger.info('Computing subset of NYTimes statistics.')
    for article in tqdm(cursor):
        sections = article['parsed_section']
        pars = [s['text'] for s in sections if s['type'] == 'paragraph']
        captions = []

        for s in sections:
            if s['type'] == 'caption':
                image_path = os.path.join('data/nytimes/images_processed',
                                          f"{s['hash']}.jpg")
                if os.path.exists(image_path):
                    captions.append(s['text'])
        if not captions:
            continue

        n_article_words += len(' '.join(pars).split())
        n_articles += 1
        n_caption_words += len(' '.join(captions).split())
        n_captions += len(captions)

        if article['pub_date'] < min_date:
            min_date = article['pub_date']
        if article['pub_date'] > max_date:
            max_date = article['pub_date']

    print('Subset of NYTimes Dataset:')
    print('No of articles:', n_articles)
    print('No of captions:', n_captions)
    print('Average article len:', n_article_words / n_articles)
    print('Average caption len:', n_caption_words / n_captions)
    print('Min date:', min_date)
    print('Max date:', max_date)
    print()


def compute_nytimes_exact_subset_statistics(nytimes, goodnews):
    cursor = goodnews.splits.find({}, no_cursor_timeout=True)
    article_ids = set({})

    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_articles = 0

    max_date = datetime(2000, 1, 1)
    min_date = datetime(2020, 1, 1)

    logger.info('Computing exact subset of NYTimes statistics.')
    for sample in tqdm(cursor):
        article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })

        if sample['article_id'] not in article_ids:
            article_ids.add(sample['article_id'])
            article = nytimes.articles.find_one({'_id': sample['article_id']})

            if article is None or 'parsed_section' not in article:
                continue

            sections = article['parsed_section']
            pars = [s['text'] for s in sections if s['type'] == 'paragraph']
            captions = []
            for s in sections:
                if s['type'] == 'caption':
                    image_path = os.path.join('data/nytimes/images_processed',
                                              f"{s['hash']}.jpg")
                    if os.path.exists(image_path):
                        captions.append(s['text'])

            if not captions:
                continue

            n_article_words += len(' '.join(pars).split())
            n_articles += 1
            n_caption_words += len(' '.join(captions).split())
            n_captions += len(captions)

            if article['pub_date'] < min_date:
                min_date = article['pub_date']
            if article['pub_date'] > max_date:
                max_date = article['pub_date']

    print('Subset of NYTimes Dataset:')
    print('No of articles:', n_articles)
    print('No of captions:', n_captions)
    print('Average article len:', n_article_words / n_articles)
    print('Average caption len:', n_caption_words / n_captions)
    print('Min date:', min_date)
    print('Max date:', max_date)
    print()


def compute_goodnews_stats(goodnews):
    cursor = goodnews.splits.find({})
    article_ids = set({})
    n_article_words = 0
    n_caption_words = 0
    n_captions = 0

    logger.info('Computing GoodNews statistics.')
    for sample in tqdm(cursor):
        article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })

        image_path = os.path.join(
            'data/goodnews/images_processed', f"{sample['_id']}.jpg")
        if not os.path.exists(image_path):
            continue

        if sample['article_id'] not in article_ids:
            article_ids.add(sample['article_id'])
            n_article_words += len(article['article'].strip().split())

        caption = article['images'][sample['image_index']].strip()
        n_caption_words += len(caption.split())
        n_captions += 1

    print('Subset of GoodNews Dataset:')
    print('No of articles:', len(article_ids))
    print('No of captions:', n_captions)
    print('Average article len:', n_article_words / len(article_ids))
    print('Average caption len:', n_caption_words / n_captions)
    print()


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
    nytimes = client.nytimes
    goodnews = client.goodnews

    # compute_goodnews_stats(goodnews)
    compute_nytimes_stats(nytimes)
    # compute_nytimes_subset_statistics(nytimes)
    compute_nytimes_exact_subset_statistics(nytimes, goodnews)


if __name__ == '__main__':
    main()
