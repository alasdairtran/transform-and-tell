"""Get articles from the New York Times API.

Usage:
    compute_ner_statistics_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].

"""
import pickle
from collections import Counter
from datetime import datetime

import ptvsd
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
        'host': str
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

    client = MongoClient(host=args['host'], port=27017)
    nytimes = client.nytimes

    start = datetime(2000, 1, 1)
    end = datetime(2019, 5, 1)

    type_counter = Counter()
    noun_counter = Counter()

    article_cursor = nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'pub_date': {'$gte': start, '$lt': end},
        'language': 'en',
    }, no_cursor_timeout=True, projection=['parsed_section']).batch_size(128)

    for article in tqdm(article_cursor):
        sections = article['parsed_section']
        for section in sections:
            if section['type'] == 'caption':
                get_ner_statistics(section, type_counter, noun_counter)

    counters = {
        'type': type_counter,
        'noun': noun_counter,
    }
    with open('./data/nytimes/ner_noun_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)


def get_ner_statistics(section, type_counter, noun_counter):
    entity_locs = []
    if 'named_entities' in section:
        for entity in section['named_entities']:
            type_counter.update([entity['label']])
            entity_locs.append((entity['start'], entity['end']))

    # all nouns that are not part of an entity
    if 'parts_of_speech' in section:
        for pos in section['parts_of_speech']:
            if pos['pos'] == 'NOUN' and not part_of_entity(pos, entity_locs):
                noun_counter.update([pos['text']])


def part_of_entity(pos, entity_locs):
    for start, end in entity_locs:
        if pos['start'] >= start and pos['end'] <= end:
            return True
    return False


if __name__ == '__main__':
    main()
