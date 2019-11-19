"""Get articles from the New York Times API.

Usage:
    get_unknown_caption_names.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]

"""
import pickle

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
        'host': str,
    })
    args = schema.validate(args)
    return args


def get_name_stats(host):
    client = MongoClient(host=host, port=27017)
    db = client.nytimes

    projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                  'parsed_section.hash', 'parsed_section.parts_of_speech',
                  'parsed_section.named_entities',
                  'image_positions', 'headline',
                  'web_url', 'n_images_with_faces']

    article_cursor = db.articles.find({
        'split': 'train',
    }, no_cursor_timeout=True, projection=projection).batch_size(128)

    results = {}
    count, total = 0, 0

    for article in tqdm(article_cursor):
        article_names = set()
        caption_names = set()

        sections = article['parsed_section']
        for section in sections:
            if section['type'] == 'paragraph':
                article_names |= get_proper_names(section)
            elif section['type'] == 'caption':
                caption_names |= get_proper_names(section)

        unknown_names = set()
        for name in caption_names:
            if name not in article_names:
                unknown_names.add(name)

        if unknown_names:
            results[article['_id']] = sorted(unknown_names)
        count += len(unknown_names)
        total += len(caption_names)

    print('Count:', count)
    print('Total:', total)
    print('No articles with unknown names:', len(results))

    with open('./data/nytimes/unknown_caption_names.pkl', 'wb') as f:
        pickle.dump(results, f)


def get_proper_names(section):
    # These name indices have the right end point excluded
    names = set()

    parts_of_speech = section['parts_of_speech']
    for pos in parts_of_speech:
        if pos['pos'] == 'PROPN':
            names.add(pos['text'])

    return names


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    get_name_stats(args['host'])


if __name__ == '__main__':
    main()
