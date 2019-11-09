"""Get articles from the New York Times API.

Usage:
    compute_data_statistics.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""
import os
import pickle
from collections import defaultdict
from datetime import datetime

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


def compute_nytimes_stats(nytimes):
    projection = ['_id', 'parsed_section.type', 'pub_date',
                  'parsed_section.parts_of_speech',
                  'parsed_section.named_entities',
                  'parsed_section.hash', 'parsed_section.text']
    cursor = nytimes.articles.find({}, projection=projection)

    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_articles = 0

    n_words = 0
    n_nouns = 0
    n_adjs = 0
    n_verbs = 0
    n_pnouns = 0
    n_propers = 0
    n_entity_words = 0
    n_ent_sents = 0
    n_person_names = 0
    n_pers_sents = 0

    max_date = datetime(2000, 1, 1)
    min_date = datetime(2020, 1, 1)

    logger.info('Computing NYTimes statistics.')
    for article in tqdm(cursor):
        sections = article['parsed_section']
        pars = [s['text'] for s in sections if s['type'] == 'paragraph']
        captions = []

        for s in sections:
            if s['type'] == 'caption':
                image_path = os.path.join('data/nytimes/images_processed',
                                          f"{s['hash']}.jpg")
                if not os.path.exists(image_path):
                    continue
                captions.append(s['text'])

                if 'parts_of_speech' in s:
                    pos = s['parts_of_speech']
                    n_words += len(pos)
                    n_nouns += len([1 for p in pos if p['pos'] == 'NOUN'])
                    n_verbs += len([1 for p in pos if p['pos'] == 'VERB'])
                    n_adjs += len([1 for p in pos if p['pos'] == 'ADJ'])
                    n_pnouns += len([1 for p in pos if p['pos'] == 'PRON'])
                    n_propers += len([1 for p in pos if p['pos'] == 'PROPN'])

                has_person = False
                if 'named_entities' in s:
                    for e in s['named_entities']:
                        n_entity_words += len(e['text'].split())
                        if e['label'] == 'PERSON':
                            n_person_names += len(e['text'].split())
                            has_person = True
                    if s['named_entities']:
                        n_ent_sents += 1
                    if has_person:
                        n_pers_sents += 1

        if not captions:
            continue

        n_article_words += len(' '.join(pars).split())
        n_articles += 1
        n_caption_words += len(' '.join(captions).split())
        n_captions += len(captions)

        if article['pub_date'] < min_date:
            min_date = article['pub_date']
        if article['pub_date'] > max_date:
            max_date = article['pub_date']

    print('Full NYTimes Dataset:')
    print('No of articles:', n_articles)
    print('No of captions:', n_captions)
    print('Average article len:', n_article_words / n_articles)
    print('Average caption len:', n_caption_words / n_captions)
    print('Min date:', min_date)
    print('Max date:', max_date)
    print(f'Words: {n_words}')
    print(f'Nouns: {n_nouns} ({n_nouns / n_words:.2%})')
    print(f'Verbs: {n_verbs} ({n_verbs / n_words:.2%})')
    print(f'Adjectives: {n_adjs} ({n_adjs / n_words:.2%})')
    print(f'Pronouns: {n_pnouns} ({n_pnouns / n_words:.2%})')
    print(f'Proper nouns: {n_propers} ({n_propers / n_words:.2%})')
    print(f'Entity words: {n_entity_words} ({n_entity_words / n_words:.2%})')
    print(f'Entity sents: {n_ent_sents} ({n_ent_sents / n_captions:.2%})')
    print(f'Person names: {n_person_names} ({n_person_names / n_words:.2%})')
    print(f'Person sents: {n_pers_sents} ({n_pers_sents / n_captions:.2%})')
    print()


def compute_nytimes_exact_subset_statistics(nytimes, goodnews):
    cursor = goodnews.splits.find(
        {}, projection=['_id', 'article_id'], no_cursor_timeout=True)
    article_ids = set({})

    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_articles = 0

    n_total_nytimes_captions = 0
    n_total_goodnews_captions = 0

    diff_articles = {}

    max_date = datetime(2000, 1, 1)
    min_date = datetime(2020, 1, 1)

    logger.info('Computing exact subset of NYTimes statistics.')
    for sample in tqdm(cursor):
        goodnews_article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        }, projection=['_id', 'images'])

        if sample['article_id'] not in article_ids:
            article_ids.add(sample['article_id'])
            article = nytimes.articles.find_one({'_id': sample['article_id']})

            if article is None or 'parsed_section' not in article:
                continue

            sections = article['parsed_section']
            pars = [s['text'] for s in sections if s['type'] == 'paragraph']

            n_new_captions = len([1 for s in sections
                                  if s['type'] == 'caption'])
            n_total_nytimes_captions += n_new_captions

            n_old_captions = len(goodnews_article['images'])
            n_total_goodnews_captions += n_old_captions

            if n_new_captions != n_old_captions:
                diff_articles[sample['article_id']] = {
                    'n_new_captions': n_new_captions,
                    'n_old_captions': n_old_captions,
                }

            captions = []
            for s in sections:
                if s['type'] == 'caption':
                    image_path = os.path.join('data/nytimes/images_processed',
                                              f"{s['hash']}.jpg")
                    if os.path.exists(image_path):
                        captions.append(s['text'])

            if not captions:
                continue

            n_article_words += len(' '.join(pars).split())
            n_articles += 1
            n_caption_words += len(' '.join(captions).split())
            n_captions += len(captions)

            if article['pub_date'] < min_date:
                min_date = article['pub_date']
            if article['pub_date'] > max_date:
                max_date = article['pub_date']

    with open('data/diff_articles.pkl', 'wb') as f:
        pickle.dump(diff_articles, f)

    print('Subset of NYTimes Dataset:')
    print('No of articles:', n_articles)
    print('No of total NYTimes captions:', n_total_nytimes_captions)
    print('No of total GoodNews captions:', n_total_goodnews_captions)
    print('Average article len:', n_article_words / n_articles)
    print('Average caption len:', n_caption_words / n_captions)
    print('Min date:', min_date)
    print('Max date:', max_date)
    print()


def compute_goodnews_stats(goodnews):
    cursor = goodnews.splits.find({})
    article_ids = set({})
    n_article_words = 0
    n_caption_words = 0
    n_captions = 0
    n_original_captions = 0
    language_counter = defaultdict(int)

    n_words = 0
    n_nouns = 0
    n_adjs = 0
    n_verbs = 0
    n_pnouns = 0
    n_propers = 0
    n_entity_words = 0
    n_ent_sents = 0
    n_person_names = 0
    n_pers_sents = 0

    projection = ['_id', 'article', 'language', 'images',
                  'caption_parts_of_speech', 'caption_ner']

    logger.info('Computing GoodNews statistics.')
    for sample in tqdm(cursor):
        n_original_captions += 1
        article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        }, projection=projection)

        image_path = os.path.join(
            'data/goodnews/images_processed', f"{sample['_id']}.jpg")
        if not os.path.exists(image_path):
            continue

        if sample['article_id'] not in article_ids:
            article_ids.add(sample['article_id'])
            n_article_words += len(article['article'].strip().split())
            language_counter[article['language']] += 1

        caption = article['images'][sample['image_index']].strip()
        n_caption_words += len(caption.split())
        n_captions += 1

        pos = article['caption_parts_of_speech'][sample['image_index']]
        n_words += len(pos)
        n_nouns += len([1 for p in pos if p['pos'] == 'NOUN'])
        n_verbs += len([1 for p in pos if p['pos'] == 'VERB'])
        n_adjs += len([1 for p in pos if p['pos'] == 'ADJ'])
        n_pnouns += len([1 for p in pos if p['pos'] == 'PRON'])
        n_propers += len([1 for p in pos if p['pos'] == 'PROPN'])

        ners = article['caption_ner'][sample['image_index']]
        has_person = False
        for e in ners:
            n_entity_words += len(e['text'].split())
            if e['label'] == 'PERSON':
                n_person_names += len(e['text'].split())
                has_person = True
        if ners:
            n_ent_sents += 1
        if has_person:
            n_pers_sents += 1

    print('Subset of GoodNews Dataset:')
    print('No of articles:', len(article_ids))
    print('No of captions:', n_captions)
    print('No of original captions:', n_original_captions)
    print('Average article len:', n_article_words / len(article_ids))
    print('Average caption len:', n_caption_words / n_captions)
    print('Language stats')
    for lang, count in language_counter.items():
        print(lang, count)
    print(f'Words: {n_words}')
    print(f'Nouns: {n_nouns} ({n_nouns / n_words:.2%})')
    print(f'Verbs: {n_verbs} ({n_verbs / n_words:.2%})')
    print(f'Adjectives: {n_adjs} ({n_adjs / n_words:.2%})')
    print(f'Pronouns: {n_pnouns} ({n_pnouns / n_words:.2%})')
    print(f'Proper nouns: {n_propers} ({n_propers / n_words:.2%})')
    print(f'Entity words: {n_entity_words} ({n_entity_words / n_words:.2%})')
    print(f'Entity sents: {n_ent_sents} ({n_ent_sents / n_captions:.2%})')
    print(f'Person names: {n_person_names} ({n_person_names / n_words:.2%})')
    print(f'Person sents: {n_pers_sents} ({n_pers_sents / n_captions:.2%})')
    print()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
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

    client = MongoClient(host='localhost', port=27017)
    nytimes = client.nytimes
    goodnews = client.goodnews

    compute_goodnews_stats(goodnews)
    compute_nytimes_stats(nytimes)
    compute_nytimes_exact_subset_statistics(nytimes, goodnews)


if __name__ == '__main__':
    main()
