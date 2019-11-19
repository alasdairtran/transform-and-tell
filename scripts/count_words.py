"""Get articles from the New York Times API.

Usage:
    count_words.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -o --out-path FILE  Output file [default: data/goodnews/counters.pkl].

"""
import os
import pickle
from collections import Counter

import ptvsd
import torch
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.data.token_indexers.roberta_indexer import to_token_ids
from tell.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'out_path': str,
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

    if os.path.exists(args['out_path']):
        logger.info(f"Path already exists: {args['out_path']}")

    client = MongoClient(host='localhost', port=27017)
    db = client.goodnews
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

    out_dict = {}
    for split in ['train', 'val', 'test']:
        sample_cursor = db.splits.find({
            'split': {'$eq': split},
        }, no_cursor_timeout=True).batch_size(128)

        caption_counter = Counter()
        context_counter = Counter()
        full_counter = Counter()

        for sample in tqdm(sample_cursor):
            # Find the corresponding article
            article = db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            })
            context = article['context'].strip()
            caption = article['images'][sample['image_index']].strip()

            context_ids = to_token_ids(context, roberta)
            caption_ids = to_token_ids(caption, roberta)

            context_counter.update(context_ids)
            caption_counter.update(caption_ids)
            full_counter.update(context_ids)
            full_counter.update(caption_ids)

        out_dict[split] = {
            'context': context_counter,
            'caption': caption_counter,
            'full': full_counter,
        }

    with open(args['out_path'], 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == '__main__':
    main()
