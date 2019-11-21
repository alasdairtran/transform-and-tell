import base64
import hashlib
from urllib.parse import urlparse
from urllib.request import urlopen

import bs4
import requests
from posixpath import normpath


class ExtractError(Exception):
    pass


def extract_article(url, selected_pos):
    response = urlopen(url, timeout=5)
    raw_html = response.read().decode('utf-8')
    try:
        parsed_sections, title = extract_text(raw_html)
    except:
        raise ExtractError(f'Error parsing the article. Pick another URL.')

    if not parsed_sections:
        raise ExtractError(f'No article text is found. Pick another URL.')

    image_positions = []
    for i, section in enumerate(parsed_sections):
        if section['type'] == 'caption':
            image_positions.append(i)
            img_response = requests.get(section['url'], stream=True)
            section['image_data'] = base64.b64encode(img_response.content)

    if not image_positions:
        raise ExtractError(
            f'No image is found in the article. Pick another URL.')

    try:
        pos = image_positions[selected_pos - 1]
    except IndexError:
        raise ExtractError(
            f'The article only has {len(image_positions)} image(s) but you '
            f'select the image at position {selected_pos}. '
            f'Pick a position less than {len(image_positions) + 1}.')

    true_caption = parsed_sections[pos]['text']
    image_url = parsed_sections[pos]['url']

    article = {
        'sections': parsed_sections,
        'image_position': pos,
        'title': title,
        'true_caption': true_caption,
        'image_url': image_url,
    }
    return article


def extract_text(html):
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
