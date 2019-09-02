"""Get articles from the New York Times API.

Usage:
    get_goodnews.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -r --root-dir DIR   Root directory of data [default: data/goodnews].
    --resume            Resume the image downloading.

"""
import json
import os
import re

import ptvsd
import requests
from bs4 import BeautifulSoup
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


def strip_html(text):
    """Remove/replace HTML elements, e.g.

    An auction at Christie&apos;s in New York.
    <b> PRECEDENT </b> The 1885 label of a drink made by the inventor of Coca-Cola.
    <strong>CHANGING TECHNOLOGY</strong>  Sandbags are the traditional solution, but Fargo is turning to newer innovations like TrapBags.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    """Remove square annotations, e.g.

    Cora Bissett and Matthew Pidgeon in "Midsummer [a play with songs]."
    “Panthers Indoctrinate The Young,” Aug. 18, 1969. [Click to read on desktop]
    "[To be titled]," 2015.

    But not all squares are bad, e.g.

    Hujar’s “Susan Sontag,” 1975, and “Fran Lebowitz [at Home in Morristown],” 1974.
    Pau Poch in “[REC]2,” about a paramilitary team in Barcelona.
    Chadwick Boseman: “A lot of times, being [a black man] in Hollywood, when you get material you’ll read it and you’ll be like, ‘That’s not us.’”
    Will Rawls performing in “I make me [sic].”

    We keep this anyway to be consistent with the original dataset.
    """
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def get_goodnews_articles(root_dir, db, resume):
    with open(os.path.join(root_dir, 'img_splits.json')) as f:
        img_splits = json.load(f)
    with open(os.path.join(root_dir, 'article_caption.json')) as f:
        article_captions = json.load(f)
    with open(os.path.join(root_dir, 'image_urls.json')) as f:
        img_urls = json.load(f)

    if not resume:
        logger.info('Inserting Good News articles.')
        for id_, article in tqdm(article_captions.items()):
            result = db.articles.find_one({'_id': id_})
            if result is None:
                article['_id'] = id_
                article['web_url'] = article['article_url']

                for idx, caption in article['images'].items():
                    caption = caption.strip()
                    article['images'][idx] = denoise_text(caption)

                context = article['article'].strip()
                try:
                    title = article['headline']['main'].strip()
                    context = title + '\n\n' + context
                except KeyError:
                    pass
                article['context'] = context

                db.articles.insert_one(article)

    if not resume:
        logger.info('Storing splits.')
        for id_, split in tqdm(img_splits.items()):
            result = db.splits.find_one({'_id': id_})
            if result is None:
                db.splits.insert_one({
                    '_id': id_,
                    'article_id': id_.split('_')[0],
                    'image_index': id_.split('_')[1],
                    'split': split,
                })

    logger.info('Downloading images.')
    for id_, links in tqdm(img_urls.items()):
        for ix, img_url in links.items():
            img_path = os.path.join(root_dir, 'images', f"{id_}_{ix}.jpg")
            if not os.path.exists(img_path):
                img_data = requests.get(img_url, stream=True).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'root_dir': os.path.exists,
        'resume': bool,
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

    root_dir = args['root_dir']
    img_dir = os.path.join(root_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Get the nytimes database
    client = MongoClient(host='localhost', port=27017)
    db = client.goodnews

    get_goodnews_articles(root_dir, db, args['resume'])


if __name__ == '__main__':
    main()
