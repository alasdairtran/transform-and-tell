"""Get articles from the New York Times API.

Usage:
    get_articles.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -r --root-dir DIR   Root directory of data [default: data/nytimes].

"""

import json
import os
import time
from itertools import product
from urllib.parse import urlparse
from urllib.request import urlopen

import ptvsd
import requests
from bs4 import BeautifulSoup
from docopt import docopt
from goose3 import Goose
from posixpath import normpath
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


def resolve_url(url):
    """
    resolve_url('http://www.example.com/foo/bar/../../baz/bux/')
    'http://www.example.com/baz/bux/'
    resolve_url('http://www.example.com/some/path/../file.ext')
    'http://www.example.com/some/file.ext'
    """

    parsed = urlparse(url)
    new_path = normpath(parsed.path)
    if parsed.path.endswith('/'):
        # Compensate for issue1707768
        new_path += '/'
    cleaned = parsed._replace(path=new_path)

    return cleaned.geturl()


def retrieve_articles(root_dir, year, month):
    in_path = os.path.join(root_dir, 'archive', f'{year}_{month:02}.json')
    out_path = os.path.join(root_dir, 'data', f'{year}_{month:02}.json')
    if os.path.exists(out_path):
        return

    month_data = {}
    with open(in_path) as f:
        content = json.load(f)
        for article in tqdm(content['response']['docs']):
            data = retrieve_article(article, root_dir)
            month_data[article['_id']] = data

    with open(out_path) as f:
        json.dump(month_data, f)


def get_soup(url):
    sauce = requests.get(url)
    soup = BeautifulSoup(sauce.content, 'html.parser')
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('noscript')]
    figcap = soup.find_all("figcaption")
    return soup, figcap


def retrieve_article(article, root_dir):
    data = {}
    url = resolve_url(article['web_url'])
    g = Goose()
    extract = g.extract(url=url)

    data['headline'] = article.get('headline', None)
    data['article_url'] = url
    data['article'] = extract.cleaned_text
    data['abstract'] = article.get('abstract', None)

    if article['multimedia']:
        data['images'] = {}
        _, figcap = get_soup(url)
        figcap = [c for c in figcap if c.text]
        for ix, cap in enumerate(figcap):
            if cap.parent.attrs.get('itemid', 0):
                img_url = resolve_url(cap.parent.attrs['itemid'])
                img_data = requests.get(img_url, stream=True).content
                img_path = os.path.join(root_dir, 'images',
                                        f"{article['_id']}_{ix}.jpg")

                with open(img_path, 'wb') as f:
                    f.write(img_data)

                text = cap.get_text().split('credit')[0]
                text = text.split('Credit')[0]
                data['images'].update({ix: text})
    return data


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'root_dir': os.path.exists,
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

    years = range(2018, 1979, -1)
    months = range(1, 13)

    root_dir = args['root_dir']
    data_dir = os.path.join(root_dir, 'data')
    img_dir = os.path.join(root_dir, 'images')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    for year, month in product(years, months):
        logger.info(f'Retrieve articles from {month}-{year}.')
        retrieve_articles(root_dir, year, month)


if __name__ == '__main__':
    main()
