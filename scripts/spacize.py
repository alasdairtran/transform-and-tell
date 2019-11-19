"""Get articles from the New York Times API.

Usage:
    spacize.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -o --out-path FILE  Output file [default: data/goodnews/annotations.pkl].

"""
import os
import pickle

import ptvsd
import spacy
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

    logger.info('Loading spacy.')
    nlp = spacy.load("en_core_web_lg", disable=['parser', 'tagger'])
    client = MongoClient(host='localhost', port=27017)
    db = client.goodnews

    article_cursor = db.articles.find(
        {}, no_cursor_timeout=True).batch_size(128)

    out_dict = {}
    if os.path.exists(args['out_path']):
        logger.info('Loading existing annotations.')
        with open(args['out_path'], 'rb') as f:
            out_dict = pickle.load(f)

    logger.info('Annotating articles.')
    for i, article in enumerate(tqdm(article_cursor)):
        if article['_id'] in out_dict:
            continue
        context = nlp(article['context'].strip())
        caption_list = list(article['images'].items())
        caption_texts = [x[1] for x in caption_list]
        caption_docs = list(nlp.pipe([c.strip() for c in caption_texts]))
        caption_dict = {}
        for d, x in zip(caption_docs, caption_list):
            caption_dict[x[0]] = d.to_bytes()
        out_dict[article['_id']] = {
            'context': context.to_bytes(),
            'captions': caption_dict
        }

        if i % 50000 == 0:
            with open(args['out_path'], 'wb') as f:
                pickle.dump(out_dict, f)

    with open(args['out_path'], 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == '__main__':
    main()
