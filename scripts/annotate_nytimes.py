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


def get_parts_of_speech(doc, article):
    parts_of_speech = []
    for tok in doc:
        pos = {
            'start': tok.idx,
            'end': tok.idx + len(tok.text),  # exclude right endpoint
            'text': tok.text,
            'pos': tok.pos_,
        }
        parts_of_speech.append(pos)

        if 'main' in article['headline']:
            section = article['headline']
            assign_pos_to_section(section, pos)

        for section in article['parsed_section']:
            assign_pos_to_section(section, pos)

    article['parts_of_speech'] = parts_of_speech


def assign_pos_to_section(section, pos):
    s = section['spacy_start']
    e = section['spacy_end']
    if pos['start'] >= s and pos['end'] <= e:
        section['parts_of_speech'].append({
            'start': pos['start'] - s,
            'end': pos['end'] - s,
            'text': pos['text'],
            'pos':  pos['pos'],
        })


def calculate_spacy_positions(article):
    title = ''
    cursor = 0
    if 'main' in article['headline']:
        title = article['headline']['main'].strip()
        article['headline']['spacy_start'] = cursor
        cursor += len(title) + 1  # newline
        article['headline']['spacy_end'] = cursor
        article['headline']['parts_of_speech'] = []

    for section in article['parsed_section']:
        text = section['text'].strip()
        section['spacy_start'] = cursor
        cursor += len(text) + 1  # newline
        section['spacy_end'] = cursor
        section['parts_of_speech'] = []


def annotate_pos(article, nlp, db):
    if 'parts_of_speech' in article['parsed_section'][0]:
        return

    calculate_spacy_positions(article)

    title = ''
    if 'main' in article['headline']:
        title = article['headline']['main'].strip()

    sections = article['parsed_section']

    paragraphs = [s['text'].strip() for s in sections]
    paragraphs = [title] + paragraphs

    combined = '\n'.join(paragraphs)

    doc = nlp(combined)
    get_parts_of_speech(doc, article)

    db.articles.find_one_and_update(
        {'_id': article['_id']}, {'$set': article})


def parse_article(article, nlp, db):
    annotate_pos(article, nlp, db)

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
