"""Get articles from the New York Times API.

Usage:
    dump_database.py [options] DUMP_PATH

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    DUMP_PATH           Dump path [default: nytimes.jsonl].

"""
import json
import os

import docopt
import ptvsd
import pymongo
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

logger = setup_logger()


def dump_text(dump_path):
    client = MongoClient(host='localhost', port=27017)
    db = client.nytimes

    articles = []

    projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                  'headline.main', 'web_url', 'pub_date', 'type_of_material',
                  'news_desk', 'abstract']

    cursor = db.articles.find({}, projection=projection).sort(
        'pub_date', pymongo.DESCENDING)

    news_desks = set()
    type_of_materials = set()

    for a in tqdm(cursor):
        sections = a['parsed_section']
        paragraphs = [s['text'].strip()
                      for s in sections if s['type'] == 'paragraph']

        article = {
            'id': a['_id'],
            'url': a['web_url'],
            'pub_date': str(a['pub_date']),
            'content': '\n'.join(paragraphs).strip(),
            'type_of_material': a['type_of_material'],
            'news_desk': a['news_desk'],
        }

        if 'main' in a['headline']:
            article['headline'] = a['headline']['main']
        if 'abstract' in a:
            article['abstract'] = a['abstract']

        articles.append(article)
        news_desks.add(a['news_desk'])
        type_of_materials.add(a['type_of_material'])

    with open(dump_path, 'a') as f:
        for a in articles:
            f.write(f'{json.dumps(a)}\n')

    # Then to compress, run this in the command line
    # gzip nytimes.jsonl
    #
    # To read the file:
    # import json, gzip
    # with gzip.open('nytimes.jsonl.gz') as f:
    #     for ln in f:
    #         obj = json.loads(ln)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        object: object,
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

    dump_text(args['dump_path'])


if __name__ == '__main__':
    main()
