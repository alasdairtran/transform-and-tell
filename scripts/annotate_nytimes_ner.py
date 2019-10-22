"""Get articles from the New York Times API.

Usage:
    annotate_nytimes_ner.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].

"""

import ptvsd
import spacy
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

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


def parse_article(article, nlp, db):
    sections = article['parsed_section']
    changed = False

    for section in sections:
        if section['type'] == 'caption':
            doc = nlp(section['text'].strip())
            section['named_entities'] = []
            for ent in doc.ents:
                changed = True
                ent_info = {
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'label': ent.label_,
                }
                section['named_entities'].append(ent_info)

    if changed:
        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


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
    db = client.nytimes

    article_cursor = db.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'language': 'en',
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Annotating articles.')
    with Parallel(n_jobs=4, backend='threading') as parallel:
        parallel(delayed(parse_article)(article, nlp, db)
                 for article in tqdm(article_cursor))


if __name__ == '__main__':
    main()
