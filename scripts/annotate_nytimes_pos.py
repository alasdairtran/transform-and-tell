"""Get articles from the New York Times API.

Usage:
    annotate_corefs.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""

import ptvsd
import spacy
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

import neuralcoref
from tell.utils import setup_logger

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


def parse_article(article, nlp, db):
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


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    logger.info('Loading spacy.')
    nlp = spacy.load("en_core_web_lg")
    neuralcoref.add_to_pipe(nlp)
    client = MongoClient(host='localhost', port=27017)
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
