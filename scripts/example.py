import os
from datetime import datetime

from PIL import Image
from pymongo import MongoClient
from tqdm import tqdm


def get_face_path(face_dir, img_hash, pos, i):
    if i == 0:
        path = os.path.join(face_dir, f"{img_hash}_{pos:02}.jpg")
    else:
        path = os.path.join(face_dir, f"{img_hash}_{pos:02}_{i + 1}.jpg")
    return path


def get_article_text(article):
    # First, we construct the article body text by joining all the
    # paragraph together.
    paragraphs = []
    sections = article['parsed_section']
    for section in sections:
        if section['type'] == 'paragraph':
            paragraphs.append(section['text'])
    article_text = '\n'.join(paragraphs)

    return article_text


def main(start, end, image_dir, face_dir):
    client = MongoClient(host='cray', port=27017)

    # To speed up the query, we select only the attributes we need
    projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                  'parsed_section.hash', 'parsed_section.parts_of_speech',
                  'parsed_section.facenet_details', 'parsed_section.named_entities',
                  'image_positions', 'headline',
                  'web_url', 'n_images_with_faces']

    article_cursor = client.nytimes.articles.find({
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True, projection=projection).batch_size(128)

    for article in tqdm(article_cursor):
        sections = article['parsed_section']
        image_positions = article['image_positions']

        article_text = get_article_text(article)

        # Next we loop through every image in the article
        for pos in image_positions:
            # Skip this if the caption is empty
            caption = sections[pos]['text'].strip()
            if not caption:
                continue

            # Load the actual image from disk
            section = sections[pos]
            image_path = os.path.join(image_dir, f"{section['hash']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                print(f'Image does not exist at {image_path}')
                continue

            # Check if there are any faces in the image
            if 'facenet_details' in section:
                # face_embeds is a list of all embeddings of faces, sorted
                # by face size. Each embedding is a list of 512 numbers.
                face_embeds = section['facenet_details']['embeddings']

                # We can also load the extracted face from disk:
                for i in range(len(face_embeds)):
                    img_hash = section['hash']
                    face_path = get_face_path(face_dir, img_hash, pos, i)
                    face = Image.open(face_path)

            # There are other things we get extract, such as named entities
            # and parts of speech
            named_entities = section['named_entities']
            parts_of_speech = section['parts_of_speech']


if __name__ == '__main__':
    split = 'test'
    if split == 'train':
        start = datetime(2000, 1, 1)
        end = datetime(2019, 5, 1)
    elif split == 'valid':
        start = datetime(2019, 5, 1)
        end = datetime(2019, 6, 1)
    elif split == 'test':
        start = datetime(2019, 6, 1)
        end = datetime(2019, 9, 1)

    image_dir = '/localdata/u4921817/projects/newser/data/nytimes/images_processed'
    face_dir = '/localdata/u4921817/projects/newser/data/nytimes/facenet'

    main(start, end, image_dir, face_dir)
