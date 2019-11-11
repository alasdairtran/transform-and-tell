"""Get articles from the New York Times API.

Usage:
    compute_metrics.py [options] FILE

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    FILE                Path to the json file.
    -c --counters PATH  Path to the word counters.
    --use_processed     Use processed captions instead of raw captions.

"""
import json
import os
import pickle
import re
import types
from collections import defaultdict

import numpy as np
import ptvsd
from docopt import docopt
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


# Patch meteor scorer. See https://github.com/tylin/coco-caption/issues/25
def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'file': os.path.exists,
        'counters': Or(None, os.path.exists),
        'use_processed': bool,
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

    with open(args['counters'], 'rb') as f:
        counters = pickle.load(f)

    full_counter = counters['context'] + counters['caption']

    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    recalls, precisions = [], []
    rare_recall, rare_recall_total = 0, 0
    rare_precision, rare_precision_total = 0, 0
    full_recall, full_recall_total = 0, 0
    full_precision, full_precision_total = 0, 0
    full_rare_recall, full_rare_recall_total = 0, 0
    full_rare_precision, full_rare_precision_total = 0, 0
    lengths, gt_lengths = [], []
    n_uniques, gt_n_uniques = [], []

    gen_ttrs, cap_ttrs = [], []
    gen_flesch, cap_flesch = [], []

    ent_counter = defaultdict(int)

    with open(args['file']) as f:
        for line in tqdm(f):
            obj = json.loads(line)
            if args['use_processed']:
                caption = obj['caption']
                obj['caption_names'] = obj['processed_caption_names']
            else:
                caption = obj['raw_caption']

            generation = obj['generation']

            if obj['caption_names']:
                recalls.append(compute_recall(obj))
            if obj['generated_names']:
                precisions.append(compute_precision(obj))

            c, t = compute_full_recall(obj)
            full_recall += c
            full_recall_total += t

            c, t = compute_full_precision(obj)
            full_precision += c
            full_precision_total += t

            c, t = compute_rare_recall(obj, counters['caption'])
            rare_recall += c
            rare_recall_total += t

            c, t = compute_rare_precision(obj, counters['caption'])
            rare_precision += c
            rare_precision_total += t

            c, t = compute_rare_recall(obj, full_counter)
            full_rare_recall += c
            full_rare_recall_total += t

            c, t = compute_rare_precision(obj, full_counter)
            full_rare_precision += c
            full_rare_precision_total += t

            # Remove punctuation
            caption = re.sub(r'[^\w\s]', '', caption)
            generation = re.sub(r'[^\w\s]', '', generation)

            lengths.append(len(generation.split()))
            gt_lengths.append(len(caption.split()))

            n_uniques.append(len(set(generation.split())))
            gt_n_uniques.append(len(set(caption.split())))

            bleu_scorer += (generation, [caption])
            rouge_score = rouge_scorer.calc_score([generation], [caption])
            rouge_scores.append(rouge_score)
            cider_scorer += (generation, [caption])

            stat = meteor_scorer._stat(generation, [caption])
            eval_line += ' ||| {}'.format(stat)
            count += 1

            gen_ttrs.append(obj['gen_np']['basic_ttr'])
            cap_ttrs.append(obj['caption_np']['basic_ttr'])
            gen_flesch.append(obj['gen_readability']['flesch_reading_ease'])
            cap_flesch.append(obj['caption_readability']
                              ['flesch_reading_ease'])

            compute_entities(obj, ent_counter)

    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(
            meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()

    final_metrics = {
        'BLEU-1': blue_score[0],
        'BLEU-2': blue_score[1],
        'BLEU-3': blue_score[2],
        'BLEU-4': blue_score[3],
        'ROUGE': rouge_score,
        'METEOR': meteor_score,
        'CIDEr': cider_score,
        'All names - recall': {
            'count': full_recall,
            'total': full_recall_total,
            'percentage': (full_recall / full_recall_total) if full_recall_total else None,
        },
        'All names - precision': {
            'count': full_precision,
            'total': full_precision_total,
            'percentage': (full_precision / full_precision_total) if full_precision_total else None,
        },
        'Caption rare names - recall': {
            'count': rare_recall,
            'total': rare_recall_total,
            'percentage': (rare_recall / rare_recall_total) if rare_recall_total else None,
        },
        'Caption rare names - precision': {
            'count': rare_precision,
            'total': rare_precision_total,
            'percentage': (rare_precision / rare_precision_total) if rare_precision_total else None,
        },
        'Article rare names - recall': {
            'count': full_rare_recall,
            'total': full_rare_recall_total,
            'percentage': (full_rare_recall / full_rare_recall_total) if full_rare_recall_total else None,
        },
        'Article rare names - precision': {
            'count': full_rare_precision,
            'total': full_rare_precision_total,
            'percentage': (full_rare_precision / full_rare_precision_total) if full_rare_precision_total else None,
        },
        'Length - generation': sum(lengths) / len(lengths),
        'Length - reference': sum(gt_lengths) / len(gt_lengths),
        'Unique words - generation': sum(n_uniques) / len(n_uniques),
        'Unique words - reference': sum(gt_n_uniques) / len(gt_n_uniques),
        'Caption TTR': sum(cap_ttrs) / len(cap_ttrs),
        'Generation TTR': sum(gen_ttrs) / len(gen_ttrs),
        'Caption Flesch Reading Ease': sum(cap_flesch) / len(cap_flesch),
        'Generation Flesch Reading Ease': sum(gen_flesch) / len(gen_flesch),
        'Entity all - recall': {
            'count': ent_counter['n_caption_ent_matches'],
            'total': ent_counter['n_caption_ents'],
            'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
        },
        'Entity all - precision': {
            'count': ent_counter['n_gen_ent_matches'],
            'total': ent_counter['n_gen_ents'],
            'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
        },
        'Entity person - recall': {
            'count': ent_counter['n_caption_person_matches'],
            'total': ent_counter['n_caption_persons'],
            'percentage': ent_counter['n_caption_person_matches'] / ent_counter['n_caption_persons'],
        },
        'Entity person - precision': {
            'count': ent_counter['n_gen_person_matches'],
            'total': ent_counter['n_gen_persons'],
            'percentage': ent_counter['n_gen_person_matches'] / ent_counter['n_gen_persons'],
        },
        'Entity GPE - recall': {
            'count': ent_counter['n_caption_gpes_matches'],
            'total': ent_counter['n_caption_gpes'],
            'percentage': ent_counter['n_caption_gpes_matches'] / ent_counter['n_caption_gpes'],
        },
        'Entity GPE - precision': {
            'count': ent_counter['n_gen_gpes_matches'],
            'total': ent_counter['n_gen_gpes'],
            'percentage': ent_counter['n_gen_gpes_matches'] / ent_counter['n_gen_gpes'],
        },
        'Entity ORG - recall': {
            'count': ent_counter['n_caption_orgs_matches'],
            'total': ent_counter['n_caption_orgs'],
            'percentage': ent_counter['n_caption_orgs_matches'] / ent_counter['n_caption_orgs'],
        },
        'Entity ORG - precision': {
            'count': ent_counter['n_gen_orgs_matches'],
            'total': ent_counter['n_gen_orgs'],
            'percentage': ent_counter['n_gen_orgs_matches'] / ent_counter['n_gen_orgs'],
        },
        'Entity DATE - recall': {
            'count': ent_counter['n_caption_date_matches'],
            'total': ent_counter['n_caption_date'],
            'percentage': ent_counter['n_caption_date_matches'] / ent_counter['n_caption_date'],
        },
        'Entity DATE - precision': {
            'count': ent_counter['n_gen_date_matches'],
            'total': ent_counter['n_gen_date'],
            'percentage': ent_counter['n_gen_date_matches'] / ent_counter['n_gen_date'],
        },
    }

    serialization_dir = os.path.dirname(args['file'])
    filename = os.path.basename(args['file']).split('.')[0]
    if args['use_processed']:
        filename += '_processed'

    output_file = os.path.join(
        serialization_dir, f'{filename}_reported_metrics.json')
    with open(output_file, 'w') as file:
        json.dump(final_metrics, file, indent=4)

    for key, metric in final_metrics.items():
        print(f"{key}: {metric}")


def compute_entities(obj, c):
    caption_entities = obj['caption_entities']
    gen_entities = obj['generated_entities']
    # context_entities = obj['context_entities']

    c['n_caption_ents'] += len(caption_entities)
    c['n_gen_ents'] += len(gen_entities)
    for ent in gen_entities:
        if contain_entity(caption_entities, ent):
            c['n_gen_ent_matches'] += 1
    for ent in caption_entities:
        if contain_entity(gen_entities, ent):
            c['n_caption_ent_matches'] += 1

    caption_persons = [e for e in caption_entities if e['label'] == 'PERSON']
    gen_persons = [e for e in gen_entities if e['label'] == 'PERSON']
    c['n_caption_persons'] += len(caption_persons)
    c['n_gen_persons'] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity(caption_persons, ent):
            c['n_gen_person_matches'] += 1
    for ent in caption_persons:
        if contain_entity(gen_persons, ent):
            c['n_caption_person_matches'] += 1

    caption_orgs = [e for e in caption_entities if e['label'] == 'ORG']
    gen_orgs = [e for e in gen_entities if e['label'] == 'ORG']
    c['n_caption_orgs'] += len(caption_orgs)
    c['n_gen_orgs'] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity(caption_orgs, ent):
            c['n_gen_orgs_matches'] += 1
    for ent in caption_orgs:
        if contain_entity(gen_orgs, ent):
            c['n_caption_orgs_matches'] += 1

    caption_gpes = [e for e in caption_entities if e['label'] == 'GPE']
    gen_gpes = [e for e in gen_entities if e['label'] == 'GPE']
    c['n_caption_gpes'] += len(caption_gpes)
    c['n_gen_gpes'] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity(caption_gpes, ent):
            c['n_gen_gpes_matches'] += 1
    for ent in caption_gpes:
        if contain_entity(gen_gpes, ent):
            c['n_caption_gpes_matches'] += 1

    caption_date = [e for e in caption_entities if e['label'] == 'DATE']
    gen_date = [e for e in gen_entities if e['label'] == 'DATE']
    c['n_caption_date'] += len(caption_date)
    c['n_gen_date'] += len(gen_date)
    for ent in gen_date:
        if contain_entity(caption_date, ent):
            c['n_gen_date_matches'] += 1
    for ent in caption_date:
        if contain_entity(gen_date, ent):
            c['n_caption_date_matches'] += 1

    return c


def contain_entity(entities, target):
    for ent in entities:
        if ent['text'] == target['text'] and ent['label'] == target['label']:
            return True
    return False


def compute_recall(obj):
    count = 0
    for name in obj['caption_names']:
        if name in obj['generated_names']:
            count += 1

    return count / len(obj['caption_names'])


def compute_precision(obj):
    count = 0
    for name in obj['generated_names']:
        if name in obj['caption_names']:
            count += 1

    return count / len(obj['generated_names'])


def compute_full_recall(obj):
    count = 0
    for name in obj['caption_names']:
        if name in obj['generated_names']:
            count += 1

    return count, len(obj['caption_names'])


def compute_full_precision(obj):
    count = 0
    for name in obj['generated_names']:
        if name in obj['caption_names']:
            count += 1

    return count, len(obj['generated_names'])


def compute_rare_recall(obj, counter):
    count = 0
    rare_names = [n for n in obj['caption_names'] if n not in counter]
    for name in rare_names:
        if name in obj['generated_names']:
            count += 1

    return count, len(rare_names)


def compute_rare_precision(obj, counter):
    count = 0
    rare_names = [n for n in obj['generated_names'] if n not in counter]
    for name in rare_names:
        if name in obj['caption_names']:
            count += 1

    return count, len(rare_names)


if __name__ == '__main__':
    main()
