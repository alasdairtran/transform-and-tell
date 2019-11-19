"""Annotate Good News with parts of speech.

Usage:
    annotate_goodnews_pos.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].

"""

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

    logger.info('Loading spacy.')
    nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
    client = MongoClient(host=args['host'], port=27017)
    db = client.goodnews

    sample_cursor = db.splits.find({}, no_cursor_timeout=True).batch_size(128)

    done_article_ids = set()
    for sample in tqdm(sample_cursor):
        if sample['article_id'] in done_article_ids:
            continue
        done_article_ids.add(sample['article_id'])

        article = db.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })

        context = article['context'].strip()
        context_doc = nlp(context)
        get_context_parts_of_speech(context_doc, article)

        article['caption_parts_of_speech'] = {}
        for idx, caption in article['images'].items():
            caption = caption.strip()
            caption_doc = nlp(caption)
            get_caption_parts_of_speech(caption_doc, article, idx)

        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


def get_context_parts_of_speech(doc, article):
    parts_of_speech = []
    for tok in doc:
        pos = {
            'start': tok.idx,
            'end': tok.idx + len(tok.text),  # exclude right endpoint
            'text': tok.text,
            'pos': tok.pos_,
        }
        parts_of_speech.append(pos)

    article['context_parts_of_speech'] = parts_of_speech


def get_caption_parts_of_speech(doc, article, idx):
    parts_of_speech = []
    for tok in doc:
        pos = {
            'start': tok.idx,
            'end': tok.idx + len(tok.text),  # exclude right endpoint
            'text': tok.text,
            'pos': tok.pos_,
        }
        parts_of_speech.append(pos)

    article['caption_parts_of_speech'][idx] = parts_of_speech


if __name__ == '__main__':
    main()
