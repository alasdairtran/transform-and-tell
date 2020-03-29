import json
import os

# os.chdir('/data/raid/transform-and-tell')

goodnews_paths = {
    r'& Biten (Avg + CtxIns)~\cite{Biten2019GoodNews}': 'data/goodnews/original_results/with article/vis_show_attend_tell_full_avg/ctx_reported_metrics.json',
    r'& Biten (TBB + AttIns)~\cite{Biten2019GoodNews}': 'data/goodnews/original_results/with article/vis_show_attend_tell_full_TBB/att_reported_metrics.json',
    r'& RoBERTa Transformer LM': 'expt/goodnews/4_no_image/serialization/generations_reported_metrics.json',
    r'& \quad + image attention ($\dagger$)': 'expt/goodnews/5_transformer_roberta/serialization/generations_reported_metrics.json',
    r'& \quad\quad + weighted RoBERTa': 'expt/goodnews/6_transformer_weighted_roberta/serialization/generations_reported_metrics.json',
    r'& \quad\quad\quad + face attention': 'expt/goodnews/8_transformer_faces/serialization/generations_reported_metrics.json',
    r'& \quad\quad\quad\quad + object attention': 'expt/goodnews/9_transformer_objects/serialization/generations_reported_metrics.json',
    r'& $\dagger$ RoBERTa $\rightarrow$ GloVe': 'expt/goodnews/2_transformer_glove/serialization/generations_reported_metrics.json',
    r'& $\dagger$ Transformer $\rightarrow$ LSTM': 'expt/goodnews/3_lstm_roberta/serialization/generations_reported_metrics.json',
    r'& $\dagger$ Use both GloVe \& LSTM': 'expt/goodnews/1_lstm_glove/serialization/generations_reported_metrics.json',
}

nytimes_paths = {
    r'& RoBERTa Transformer LM': 'expt/nytimes/4_no_image/serialization/generations_reported_metrics.json',
    r'& \quad + image attention ($\dagger$)': 'expt/nytimes/5_transformer_roberta/serialization/generations_reported_metrics.json',
    r'& \quad\quad + weighted RoBERTa': 'expt/nytimes/6_transformer_weighted_roberta/serialization/generations_reported_metrics.json',
    r'& \quad\quad\quad + location-aware': 'expt/nytimes/7_transformer_location_aware/serialization/generations_reported_metrics.json',
    r'& \quad\quad\quad\quad + face attention': 'expt/nytimes/8_transformer_faces/serialization/generations_reported_metrics.json',
    r'& \quad\quad\quad\quad\quad + object attention': 'expt/nytimes/9_transformer_objects/serialization/generations_reported_metrics.json',
    r'& $\dagger$ RoBERTa $\rightarrow$ GloVe': 'expt/nytimes/2_transformer_glove/serialization/generations_reported_metrics.json',
    r'& $\dagger$ Transformer $\rightarrow$ LSTM': 'expt/nytimes/3_lstm_roberta/serialization/generations_reported_metrics.json',
    r'& $\dagger$ Use both GloVe \& LSTM': 'expt/nytimes/1_lstm_glove/serialization/generations_reported_metrics.json',
}


def display(number, m=100, sf=3, end=' & '):
    rounded_str = '{:.3g}'.format(number * m)
    if rounded_str == '0':
        out = '0'
    elif '.' not in rounded_str and len(rounded_str) == 1:
        out = '{:2}'.format(float(rounded_str))
    elif '.' not in rounded_str and len(rounded_str) == 2:
        out = '{:1}'.format(float(rounded_str))
    elif rounded_str[1] == '.':
        out = '{:.2f}'.format(float(rounded_str))
    elif rounded_str[2] == '.':
        out = '{:.1f}'.format(float(rounded_str))
    else:
        raise

    print(out, end=end)


for i, (k, path) in enumerate(goodnews_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['BLEU-4'])
    display(o['ROUGE'])
    display(o['CIDEr'])
    display(o['Entity all - precision']['percentage'])
    display(o['Entity all - recall']['percentage'])
    display(o['Entity person - precision']['percentage'])
    display(o['Entity person - recall']['percentage'])
    display(o['Caption rare names - precision']['percentage'])
    display(o['Caption rare names - recall']['percentage'], end=r' \\ ')

    if i == 1:
        print()
        print(r'\cmidrule{2-11}')
    if i == 6:
        print()
        print(r'\cmidrule{2-11}')

    print()

for i, (k, path) in enumerate(nytimes_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['BLEU-4'])
    display(o['ROUGE'])
    display(o['CIDEr'])
    display(o['Entity all - precision']['percentage'])
    display(o['Entity all - recall']['percentage'])
    display(o['Entity person - precision']['percentage'])
    display(o['Entity person - recall']['percentage'])
    display(o['Caption rare names - precision']['percentage'])
    display(o['Caption rare names - recall']['percentage'], end=r' \\ ')

    if i == 5:
        print()
        print(r'\cmidrule{2-11}')

    print()


for i, (k, path) in enumerate(goodnews_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['BLEU-1'])
    display(o['BLEU-2'])
    display(o['BLEU-3'])
    display(o['BLEU-4'])
    display(o['ROUGE'])
    display(o['METEOR'])
    display(o['CIDEr'], end=r' \\ ')

    if i == 1:
        print()
        print(r'\cmidrule{2-9}')
    if i == 6:
        print()
        print(r'\cmidrule{2-9}')

    print()


for i, (k, path) in enumerate(nytimes_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['BLEU-1'])
    display(o['BLEU-2'])
    display(o['BLEU-3'])
    display(o['BLEU-4'])
    display(o['ROUGE'])
    display(o['METEOR'])
    display(o['CIDEr'], end=r' \\ ')

    if i == 5:
        print()
        print(r'\cmidrule{2-9}')

    print()

for i, (k, path) in enumerate(goodnews_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['All names - precision']['percentage'])
    display(o['All names - recall']['percentage'])
    display(o['Article rare names - precision']['percentage'])
    display(o['Article rare names - recall']['percentage'])
    display(o['Length - generation'], m=1)
    display(o['Generation TTR'])
    display(o['Generation Flesch Reading Ease'], m=1, end=r' \\ ')

    if i == 1:
        print()
        print(r'\cmidrule{2-9}')
    if i == 6:
        print()
        print(r'\cmidrule{2-9}')

    print()

for i, (k, path) in enumerate(nytimes_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['All names - precision']['percentage'])
    display(o['All names - recall']['percentage'])
    display(o['Article rare names - precision']['percentage'])
    display(o['Article rare names - recall']['percentage'])
    display(o['Length - generation'], m=1)
    display(o['Generation TTR'])
    display(o['Generation Flesch Reading Ease'], m=1, end=r' \\ ')

    if i == 5:
        print()
        print(r'\cmidrule{2-9}')

    print()

for i, (k, path) in enumerate(goodnews_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['Entity GPE - precision']['percentage'])
    display(o['Entity GPE - recall']['percentage'])
    display(o['Entity ORG - precision']['percentage'])
    display(o['Entity ORG - recall']['percentage'])
    display(o['Entity DATE - precision']['percentage'])
    display(o['Entity DATE - recall']['percentage'], end=r' \\ ')

    if i == 1:
        print()
        print(r'\cmidrule{2-8}')
    if i == 6:
        print()
        print(r'\cmidrule{2-8}')

    print()


for i, (k, path) in enumerate(nytimes_paths.items()):
    with open(path) as f:
        o = json.load(f)

    print(k, end=' & ')
    display(o['Entity GPE - precision']['percentage'])
    display(o['Entity GPE - recall']['percentage'])
    display(o['Entity ORG - precision']['percentage'])
    display(o['Entity ORG - recall']['percentage'])
    display(o['Entity DATE - precision']['percentage'])
    display(o['Entity DATE - recall']['percentage'], end=r' \\ ')

    if i == 5:
        print()
        print(r'\cmidrule{2-8}')

    print()
