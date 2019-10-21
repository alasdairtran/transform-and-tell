import argparse
import json
import operator
import os
import re
from collections import defaultdict, deque

import numpy as np
import ptvsd
import spacy
import stop_words
import tqdm
from nltk.tokenize import word_tokenize
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# from pycocoevalcap.spice.spice import Spice


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "Spice")
    ]
    final_scores = {}
    all_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
            for m, s in zip(method, scores):
                all_scores[m] = s
        else:
            final_scores[method] = score
            all_scores[method] = scores

    return final_scores, all_scores


def evaluate(ref, cand, get_scores=True):
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    truth = {}
    for i, caption in enumerate(ref):
        truth[i] = [caption]

    # compute bleu score
    final_scores = score(truth, hypo)

    #     print out scores
    print('Bleu_1:\t ;', final_scores[0]['Bleu_1'])
    print('Bleu_2:\t ;', final_scores[0]['Bleu_2'])
    print('Bleu_3:\t ;', final_scores[0]['Bleu_3'])
    print('Bleu_4:\t ;', final_scores[0]['Bleu_4'])
    print('METEOR:\t ;', final_scores[0]['METEOR'])
    print('ROUGE_L: ;', final_scores[0]['ROUGE_L'])
    print('CIDEr:\t ;', final_scores[0]['CIDEr'])
    # print('Spice:\t', final_scores[0]['Spice'])

    if get_scores:
        return final_scores


def organize_ner(ner):
    new = defaultdict(list)
    for k, v in ner.items():
        value = ' '.join(k.split())
        if value not in stopwords:
            new[v].append(value)
    return new


def fill_random(cap, ner_dict):
    assert cap != list
    filled = []
    for c in cap:
        if c.split('_')[0] in named_entities and c.isupper():
            ent = c.split('_')[0]
            if ner_dict[ent]:
                ner = np.random.choice(ner_dict[ent])
                filled.append(ner)
            else:
                filled.append(c)
        else:
            filled.append(c)

    return filled


def rank_sentences(cap, sent):
    # make them unicode, spacy accepts only unicode
    cap = str(cap)
    sent = [str(s) for s in sent]
    # feed them to spacy to get the vectors
    cap = nlp(cap)
    list_sent = [nlp(s) for s in sent]
    compare = [s.similarity(cap) for s in list_sent]
    # we sort the article sentences according to their similarity to produced caption
    similarity = sorted([(s, c) for s, c in zip(
        list_sent, compare)], key=lambda x: x[1], reverse=True)
    return similarity


def ner_finder(ranked_sen, score_sen, word):
    for sen, sc in zip(ranked_sen, score_sen):
        beg = sen.find(word)
        if beg is not -1:
            end = beg + len(word)
            return sen[beg:end], sc
    else:
        return float('-inf'), float('-inf')


def fill_word2vec(cap, ner_dict, ner_articles, return_ners=False):
    assert cap != list
    filled = []
    similarity = rank_sentences(' '.join(cap), ner_articles)
    ranked_sen = [s[0].text for s in similarity]
    score_sen = [s[1] for s in similarity]
    if return_ners:
        ners = []

    new = {}
    for key, values in ner_dict.items():
        temp = {}
        for word in values:
            found, sc1 = ner_finder(
                ranked_sen, score_sen, re.sub('[^A-Za-z0-9]+', ' ', word))
            found2, sc2 = ner_finder(ranked_sen, score_sen, word)
            if found:
                temp[word] = sc1
            elif ner_finder(ranked_sen, score_sen, word):
                temp[word] = sc2
            else:
                temp[word] = 0
        new[key] = temp
    new = {k: deque([i for i, _ in sorted(v.items(), key=operator.itemgetter(1), reverse=True)]) for k, v in
           new.items()}

    for c in cap:
        if c.split('_')[0] in named_entities and c.isupper():
            ent = c.split('_')[0]
            if ner_dict[ent]:
                ner = new[ent].popleft()
                # append it again, we might need to reuse some entites.
                new[ent].append(ner)
                filled.append(ner)
                if return_ners:
                    ners.append((ner, ent))
            else:
                filled.append(c)
        else:
            filled.append(c)
    if return_ners:
        return filled, ners
    else:
        return filled


def insert_word(ner_test, sen_att, ix, ner_dict, return_ner=False):
    if ner_test in named_entities:
        for ii in sen_att[ix]:
            if ii < len(article['sentence']):
                art_sen = article['sentence'][ii]
                temp = [(art_sen.find(ner), ner)
                        for ner in ner_dict[ner_test] if art_sen.find(ner) != -1]

                temp = sorted(temp, key=lambda x: x[0])
                if temp and return_ner:
                    return temp[0][1], ner_test
                if temp:
                    return temp[0][1], None
        else:
            return ner_test, None
    else:
        return ner_test, None


def insert(cap, sen_att, ner_dict, return_ners=False):
    new_sen = ''
    words = []
    if return_ners:
        ners = []

    for ix, c in enumerate(cap):
        ner_test = c.split('_')[0]
        word, ner = insert_word(ner_test, sen_att, ix, ner_dict, return_ners)
        if ner:
            ners.append((word, ner))
        words.append(word)
    #         new_sen += ' ' +
    if return_ners:
        return ' '.join(words), ners
    else:
        return ' '.join(words)


def get_proper_nouns(text, nlp):
    doc = nlp(text)
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--output', type=str, default='./data/goodnews/original_results/with article/vis_show_attend_tell_full_wavg.json',
                        help='path to model to evaluate')
    parser.add_argument('--eval_file', type=str, default='./expt/2c_transformer_lr_newlines/serialization/generations.jsonl',
                        help='Reference generation file, used to find common test examples.')
    parser.add_argument('--insertion_method', type=list, default=['ctx', 'rand', 'att'],
                        help='rand: random insertion, ctx: context/word2vec/glove insertion, att: attention insertion')
    parser.add_argument('--dump', type=bool, default=True,
                        help='Save the inserted captions in a json file')
    parser.add_argument('--ptvsd', type=bool, default=False,
                        help='Debug mode')
    opt = parser.parse_args()

    if opt.ptvsd:
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    print('Loading data.')
    test_compact = open_json('./data/goodnews/original_results/test.json')
    article_dataset = open_json(
        './data/goodnews/original_results/article.json')
    stopwords = stop_words.get_stop_words('en')
    named_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE',
                      'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

    raw_captions = {}
    with open(opt.eval_file) as f:
        for line in f:
            obj = json.loads(line)
            image_name = os.path.basename(obj['image_path'])
            image_id = image_name[:-4]
            raw_captions[image_id] = obj['raw_caption']

    print('Loading spacy model.')
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

    print('Starting the insertion process.')

    output = open_json(opt.output)
    result_dir = os.path.dirname(opt.output)
    expt_name = os.path.basename(opt.output).split('.')[0]
    dump_dir = os.path.join(result_dir, expt_name)
    os.makedirs(dump_dir, exist_ok=True)

    id_to_key = {h['image_id']: h['image_path'].split(
        '/')[1].split('_')[0] for h in output}
    id_to_index = {h['cocoid']: i for i, h in enumerate(test_compact)}
    ref, image_ids = [], []
    for h in tqdm.tqdm(output):
        imgId = h['image_id']
        index = id_to_index[imgId]
        ref.append(test_compact[index]['sentences_full'][0]['raw'])
        image_ids.append(test_compact[index]['imgid'])

    for method in opt.insertion_method:
        print('Inserting with method:', method)
        hypo = []
        if method == 'att':
            att_sen = []
        for h in tqdm.tqdm(output):
            imgId = h['image_id']
            #         cap = compact_NE(h['caption'])
            cap = word_tokenize(h['caption'])
            key = id_to_key[imgId]
            # index = id_to_index[imgId]
            # ref.append(test_compact[index]['sentences_full'][0]['raw'])

            ner_articles = article_dataset[key]['sentence_ner']
            ner_dict = article_dataset[key]['ner']
            ner_dict = organize_ner(ner_dict)

            # fill the caption with named entities
            if method == 'ctx':
                cap = fill_word2vec(cap, ner_dict, ner_articles)
                cap = ' '.join(cap)
                hypo.append(' '.join(cap.split()))
            elif method == 'rand':
                cap = fill_random(cap, ner_dict)
                cap = ' '.join(cap)
                hypo.append(' '.join(cap.split()))
            elif method == 'att':
                sen_att = np.array(h['sen_att']).squeeze(axis=2)
                sorted_sen_att = [s.argsort()[-55:][::-1] for s in sen_att]
                att_sen.append(sorted_sen_att)

                article = article_dataset[key]
                index = id_to_index[imgId]
                ner_dict = article_dataset[key]['ner']
                ner_dict = organize_ner(ner_dict)
                sen, name = insert(cap, sorted_sen_att, ner_dict, True)
                hypo.append(sen)

        # retrieve the reference sentences
        if opt.dump:
            dump_path = os.path.join(dump_dir, '%s.json' % method)
            for r, h, i in zip(ref, hypo, image_ids):
                if i not in raw_captions:
                    continue
                obj = {
                    'caption': r,
                    'raw_caption': raw_captions[i],
                    'generation': h,
                    'image_id': i,
                    'caption_names': get_proper_nouns(raw_captions[i], nlp),
                    'processed_caption_names': get_proper_nouns(r, nlp),
                    'generated_names': get_proper_nouns(h, nlp),
                }
                with open(dump_path, 'a') as f:
                    f.write(f'{json.dumps(obj)}\n')
        print('Insertion Method: %s' % method)
        sc, scs = evaluate(ref, hypo)
