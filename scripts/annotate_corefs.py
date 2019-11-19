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


def get_clusters(doc, caption_len):
    # We only care about coref clusters whose main span is inside the caption.
    kept_clusters = []
    if not doc._.has_coref:
        return kept_clusters

    for cluster in doc._.coref_clusters:
        main_span = cluster.main
        start = main_span.start_char
        end = main_span.end_char

        if start >= caption_len:
            continue

        kept_cluster = {
            'main': main_span.text,
            'start': start,
            'end': end,
            'pos': main_span[-1].pos_,
            'caption_corefs': [],
            'article_corefs': []}

        for coref in cluster.mentions:
            s = coref.start_char
            e = coref.end_char
            if e <= caption_len:
                kept_cluster['caption_corefs'].append({
                    'start': s,
                    'end': e,
                    'text': coref.text,
                    'pos': coref[-1].pos_,
                })
            elif s >= caption_len:
                kept_cluster['article_corefs'].append({
                    'start': s - caption_len,
                    'end': e - caption_len,
                    'text': coref.text,
                    'pos': coref[-1].pos_,
                })

        kept_clusters.append(kept_cluster)

    return kept_clusters


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
    db = client.goodnews

    sample_cursor = db.splits.find(
        {}, no_cursor_timeout=True).batch_size(128)

    logger.info('Annotating articles.')
    for sample in tqdm(sample_cursor):
        result = db.corefs.find_one({'_id': sample['_id']})
        if result is not None:
            continue

        article = db.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })
        caption = article['images'][sample['image_index']]
        combined = f"{caption.strip()}\n\n{article['context'].strip()}"
        caption_len = len(caption.strip()) + 2  # 2 new lines
        doc = nlp(combined)
        clusters = get_clusters(doc, caption_len)

        db.corefs.insert_one({
            '_id': sample['_id'],
            'clusters': clusters,
        })


if __name__ == '__main__':
    main()
