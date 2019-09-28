"""Get articles from the New York Times API.

Usage:
    annotate_corefs.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""

import ptvsd
import spacy
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

import neuralcoref
from newser.utils import setup_logger

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


def get_clusters(doc, article):
    # We only care about coref clusters whose main span is inside the caption.
    if not doc._.has_coref:
        article['coref_clusters'] = []
        return

    kept_clusters = []

    for i, cluster in enumerate(doc._.coref_clusters):
        main_span = cluster.main
        start = main_span.start_char
        end = main_span.end_char

        kept_cluster = {
            'main': {
                'text': main_span.text,
                'start': start,
                'end': end,
                'pos': [span.pos_ for span in main_span],
            },
            'mentions': [],
        }

        assign_coref(article, kept_cluster['main'], i, 'main', 0)

        for j, coref in enumerate(cluster.mentions):
            s = coref.start_char
            e = coref.end_char
            if s == start and e == end:
                continue

            mention = {
                'text': coref.text,       
                'start': s,
                'end': e,
                'pos': [span.pos_ for span in main_span],
            }

            kept_cluster['mentions'].append(mention)

            assign_coref(article, mention, i, 'mention', j)

        kept_clusters.append(kept_cluster)

    
    article['coref_clusters'] = kept_clusters


def assign_coref(article, coref, i, kind, mention_idx):
    if 'main' in article['headline']:
        section = article['headline']
        assign_coref_to_section(section, coref, i, kind, mention_idx)

    for section in article['parsed_section']:
        assign_coref_to_section(section, coref, i, kind, mention_idx)
        

def assign_coref_to_section(section, coref, i, kind, mention_idx):
    s = section['spacy_start']
    e = section['spacy_end']
    if coref['start'] >= s and coref['end'] <= e:
        section['corefs'].append({
            'cluster_idx': i,
            'kind': kind,
            'mention_index': mention_idx,
            'text': coref['text'],
            'start': coref['start'] - section['spacy_start'],
            'end': coref['end'] - section['spacy_start'],
            'pos': coref['pos']
        })

def calculate_spacy_positions(article):
    title = ''
    cursor = 0
    if 'main' in article['headline']:
        title = article['headline']['main'].strip()
        article['headline']['spacy_start'] = cursor
        cursor += len(title) + 1 # newline
        article['headline']['spacy_end'] = cursor
        article['headline']['corefs'] = []

    for section in article['parsed_section']:
        text = section['text'].strip()
        section['spacy_start'] = cursor
        cursor += len(text) + 1 # newline
        section['spacy_end'] = cursor
        section['corefs'] = []

    

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
            'parsed': True, # article body is parsed into paragraphs
            'n_images': {'$gt': 0}, # at least one image is present
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Annotating articles.')
    for article in tqdm(article_cursor):
        # if 'coref_clusters' in article:
        #     continue

        calculate_spacy_positions(article)

        title = ''
        if 'main' in article['headline']:
            title = article['headline']['main'].strip()

        sections = article['parsed_section']

        paragraphs = [s['text'].strip() for s in sections]
        paragraphs = [title] + paragraphs

        combined = '\n'.join(paragraphs)

        doc = nlp(combined)
        get_clusters(doc, article)

        db.articles.find_one_and_update({'_id': article['_id']}, {'$set': article})


if __name__ == '__main__':
    main()
