"""Get articles from the New York Times API.

Usage:
    annotate_nytimes_ner.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].
    -b --batch INT      Batch number [default: 1]

"""

import functools
from datetime import datetime
from multiprocessing import Pool

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
        'batch': Use(int),
    })
    args = schema.validate(args)
    return args


def parse_article(article, nlp, db):
    sections = article['parsed_section']
    changed = False

    if 'main' in article['headline'] and 'named_entities' not in article['headline']:
        section = article['headline']
        title = section['main'].strip()
        doc = nlp(title)
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

    for section in sections:
        if 'named_entities' not in section:
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


def annotate_with_host(host, period):
    start, end = period
    nlp = spacy.load("en_core_web_lg")
    client = MongoClient(host=host, port=27017)
    db = client.nytimes

    article_cursor = db.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        parse_article(article, nlp, db)


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

    annotate = functools.partial(annotate_with_host, args['host'])

    pool = Pool(processes=11)
    pool.map(annotate, periods)


if __name__ == '__main__':
    main()
