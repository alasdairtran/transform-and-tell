import base64
import hashlib
from urllib.parse import urlparse
from urllib.request import urlopen

import bs4
import requests
from posixpath import normpath


class ExtractError(Exception):
    pass


def get_urls(url):
    response = urlopen(url, timeout=5)
    raw_html = response.read().decode('utf-8')
    try:
        parsed_sections, title = extract_text(raw_html)
    except:
        raise ExtractError(f'Error parsing the article. Pick another URL.')

    if not parsed_sections:
        raise ExtractError(f'No article text is found. Pick another URL.')

    image_urls = []
    for section in parsed_sections:
        if section['type'] == 'caption':
            img_response = requests.get(section['url'], stream=True)
            section['image_data'] = str(base64.b64encode(img_response.content),
                                        'utf-8')
            image_urls.append(section['url'])

    if not image_urls:
        raise ExtractError(f'No image is found in the article. '
                           f'Pick another URL.')

    output = {
        'sections': parsed_sections,
        'title': title,
        'image_urls': image_urls,
    }
    return output


def extract_article(sections, title, selected_pos):
    positions = [i for i, s in enumerate(sections) if s['type'] == 'caption']
    pos = positions[selected_pos]

    true_caption = sections[pos]['text']
    image_url = sections[pos]['url']

    article = {
        'sections': sections,
        'image_position': pos,
        'title': title,
        'true_caption': true_caption,
        'image_url': image_url,
    }
    return article


def extract_text(html):
    # Before November 2019, we could just use BeautifulSoup to extract all
    # the image captions in the article. But since November 2019, it seems
    # that The New York Times starts to use more Javascript to insert images
    # into the page, so we can only extract the top image from now on.
    soup = bs4.BeautifulSoup(html, 'html.parser')

    title = soup.find('h1').text.strip()

    # Newer articles use StoryBodyCompanionColumn
    if soup.find('article') and soup.find('article').find_all('div', {'class': 'StoryBodyCompanionColumn'}):
        return extract_text_new(soup), title

    # Older articles use story-body
    elif soup.find_all('p', {'class': 'story-body-text'}):
        return extract_text_old(soup), title

    return []


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
                    caption_text = caption.text.strip()
                else:
                    caption_text = ''
                url = resolve_url(part.parent.attrs['itemid'])
                sections.append({
                    'type': 'caption',
                    'order': i,
                    'text': caption_text,
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
                    caption_text = caption.text.strip()
                else:
                    caption_text = ''
                url = resolve_url(part.parent.attrs['itemid'])
                sections.append({
                    'type': 'caption',
                    'order': i,
                    'text': caption_text,
                    'url': url,
                    'hash': hashlib.sha256(url.encode('utf-8')).hexdigest(),
                })
                i += 1

    return sections


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
