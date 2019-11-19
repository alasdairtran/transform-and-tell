"""Get articles from the New York Times API.

Usage:
    get_articles.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -r --root-dir DIR   Root directory of data [default: data/nytimes].

"""
import hashlib
import json
import os
import socket
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import bs4
import ptvsd
import pymongo
import requests
from docopt import docopt
from joblib import Parallel, delayed
from posixpath import normpath
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from langdetect import detect
from tell.utils import setup_logger

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


def get_tags(d, params):
    # See https://stackoverflow.com/a/57683816/3790116
    if any((lambda x: b in x if a == 'class' else b == x)(d.attrs.get(a, [])) for a, b in params.get(d.name, {}).items()):
        yield d
    for i in filter(lambda x: x != '\n' and not isinstance(x, bs4.element.NavigableString), d.contents):
        yield from get_tags(i, params)


def extract_text_new(soup):
    # For articles between 2013 and 2019
    sections = []
    article_node = soup.find('article')

    params = {
        'div': {'class': 'StoryBodyCompanionColumn'},
        'figcaption': {'itemprop': 'caption description'},
    }

    article_parts = get_tags(article_node, params)
    i = 0

    for part in article_parts:
        if part.name == 'div':
            paragraphs = part.find_all(['p', 'h2'])
            for p in paragraphs:
                sections.append({'type': 'paragraph', 'text': p.text.strip()})
        elif part.name == 'figcaption':
            if part.parent.attrs.get('itemid', 0):
                caption = part.find('span', {'class': 'e13ogyst0'})
                if caption:
                    url = resolve_url(part.parent.attrs['itemid'])
                    sections.append({
                        'type': 'caption',
                        'order': i,
                        'text': caption.text.strip(),
                        'url': url,
                        'hash': hashlib.sha256(url.encode('utf-8')).hexdigest(),
                    })
                    i += 1

    return sections


def extract_text_old(soup):
    # For articles in 2012 and earlier
    sections = []

    params = {
        'p': {'class': 'story-body-text'},
        'figcaption': {'itemprop': 'caption description'},
        'span': {'class': 'caption-text'},
    }

    article_parts = get_tags(soup, params)
    i = 0
    for part in article_parts:
        if part.name == 'p':
            sections.append({'type': 'paragraph', 'text': part.text.strip()})
        elif part.name == 'figcaption':
            if part.parent.attrs.get('itemid', 0):
                caption = part.find('span', {'class': 'caption-text'})
                if caption:
                    url = resolve_url(part.parent.attrs['itemid'])
                    sections.append({
                        'type': 'caption',
                        'order': i,
                        'text': caption.text.strip(),
                        'url': url,
                        'hash': hashlib.sha256(url.encode('utf-8')).hexdigest(),
                    })
                    i += 1

    return sections


def extract_text(html):
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # Newer articles use StoryBodyCompanionColumn
    if soup.find('article') and soup.find('article').find_all('div', {'class': 'StoryBodyCompanionColumn'}):
        return extract_text_new(soup)

    # Older articles use story-body
    elif soup.find_all('p', {'class': 'story-body-text'}):
        return extract_text_old(soup)

    return []


def retrieve_articles(root_dir, year, month, db):
    result = db.scraping.find_one({'year': year, 'month': month})
    if result is not None:
        return

    in_path = os.path.join(root_dir, 'archive', f'{year}_{month:02}.json')
    with open(in_path) as f:
        content = json.load(f)
        for article in tqdm(content['response']['docs'], desc=f'{year}-{month:02}'):
            retrieve_article(article, root_dir, db)

    db.scraping.insert_one({'year': year, 'month': month})


def retrieve_article(article, root_dir, db):
    if article['_id'].startswith('nyt://article/'):
        article['_id'] = article['_id'][14:]

    result = db.source.find_one({'_id': article['_id']})
    if result is not None:
        return

    data = article
    data['scraped'] = False
    data['parsed'] = False
    data['error'] = False
    data['pub_date'] = datetime.strptime(article['pub_date'],
                                         '%Y-%m-%dT%H:%M:%S%z')

    if not article['web_url']:
        return

    url = resolve_url(article['web_url'])
    for i in range(10):
        try:
            response = urlopen(url, timeout=20)
            break
        except (ValueError, HTTPError):
            # ValueError: unknown url type: '/interactive/2018/12/05/business/05Markets.html'
            # urllib.error.HTTPError: HTTP Error 404: Not Found
            return
        except (URLError, ConnectionResetError):
            time.sleep(60)
            continue
        except socket.timeout:
            pass
        # urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
        return

    data['web_url'] = url
    try:
        raw_html = response.read().decode('utf-8')
    except UnicodeDecodeError:
        return

    raw_data = {
        '_id': article['_id'],
        'raw_html': raw_html,
    }

    parsed_sections = extract_text(raw_html)
    data['parsed_section'] = parsed_sections

    text_list = [sec['text'] for sec in article['parsed_section']]
    text = '\n'.join(text_list)
    data['language'] = detect(text)

    if parsed_sections:
        image_positions = []
        for i, section in enumerate(parsed_sections):
            if section['type'] == 'caption':
                image_positions.append(i)
                img_path = os.path.join(root_dir, 'images',
                                        f"{section['hash']}.jpg")
                if not os.path.exists(img_path):
                    try:
                        img_response = requests.get(
                            section['url'], stream=True)
                        img_data = img_response.content
                        with open(img_path, 'wb') as f:
                            f.write(img_data)

                        db.images.update_one(
                            {'_id': section['hash']},
                            {'$push': {'captions': {
                                'id': article['_id'],
                                'caption': section['text'],
                            }}}, upsert=True)

                    except requests.exceptions.MissingSchema:
                        section['downloaded'] = False
                    else:
                        section['downloaded'] = True

        data['parsed'] = True
        article['image_positions'] = image_positions
        article['n_images'] = len(image_positions)
    else:
        article['n_images'] = 0

    data['scraped'] = True

    db.source.insert_one({'_id': raw_data['_id']}, {'$set': raw_data})

    if not article['parsed'] or article['n_images'] == 0 or article['language'] != 'en':
        db.text_articles.insert_one(article)
    else:
        db.articles.insert_one({'_id': article['_id']}, {'$set': data})


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


def month_year_iter(end_month, end_year, start_month, start_year):
    # The first month is excluded
    ym_start = 12 * start_year + start_month - 1
    ym_end = 12 * end_year + end_month - 1
    for ym in range(ym_end, ym_start, -1):
        y, m = divmod(ym, 12)
        yield y, m + 1


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
    db = client.nytimes

    with Parallel(n_jobs=12, backend='threading') as parallel:
        parallel(delayed(retrieve_articles)(root_dir, year, month, db)
                 for year, month in month_year_iter(8, 2019, 12, 2003))

    start = datetime(2019, 6, 1)
    end = datetime(2019, 9, 1)
    article_cursor = db.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)
    for article in tqdm(article_cursor):
        article['split'] = 'test'
        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})

    start = datetime(2000, 1, 1)
    end = datetime(2019, 5, 1)
    article_cursor = db.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)
    for article in tqdm(article_cursor):
        article['split'] = 'train'
        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})

    start = datetime(2019, 5, 1)
    end = datetime(2019, 6, 1)
    article_cursor = db.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)
    for article in tqdm(article_cursor):
        article['split'] = 'valid'
        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})

    # Build indices
    logger.info('Building indices')
    db.articles.create_index([
        ('pub_date', pymongo.DESCENDING),
    ])

    db.articles.create_index([
        ('pub_date', pymongo.DESCENDING),
        ('_id', pymongo.ASCENDING),
    ])

    db.articles.create_index([
        ('split', pymongo.ASCENDING),
        ('_id', pymongo.ASCENDING),
    ])

    db.articles.create_index([
        ('n_images', pymongo.ASCENDING),
        ('n_images_with_faces', pymongo.ASCENDING),
        ('pub_date', pymongo.DESCENDING),
    ])


if __name__ == '__main__':
    main()
