"""Annotate Good News with parts of speech.

Usage:
    annotate_goodnews_ner.py [options]

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
    nlp = spacy.load("en_core_web_lg")
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

        changed = False
        if 'caption_ner' not in article:
            changed = True
            article['caption_ner'] = {}
            for idx, caption in article['images'].items():
                caption = caption.strip()
                caption_doc = nlp(caption)
                get_caption_ner(caption_doc, article, idx)

        if 'context_ner' not in article:
            changed = True
            context = article['context'].strip()
            context_doc = nlp(context)
            get_context_ner(context_doc, article)

        if changed:
            db.articles.find_one_and_update(
                {'_id': article['_id']}, {'$set': article})


def get_caption_ner(doc, article, idx):
    ner = []
    for ent in doc.ents:
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }
        ner.append(ent_info)

    article['caption_ner'][idx] = ner


def get_context_ner(doc, article):
    ner = []
    for ent in doc.ents:
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }
        ner.append(ent_info)

    article['context_ner'] = ner


if __name__ == '__main__':
    main()
