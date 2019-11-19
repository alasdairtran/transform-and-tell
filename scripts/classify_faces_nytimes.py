"""Get articles from the New York Times API.

Usage:
    classify_faces_nytimes.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      Mongo host name [default: localhost]
    -f --face-dir DIR   Image directory [default: ./data/nytimes/facenet].

"""
import os
import pickle
from collections import Counter

import numpy as np
import ptvsd
from docopt import docopt
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'host': str,
        'face_dir': str,
    })
    args = schema.validate(args)
    return args


def classify_face(face_path, vggface, counter):
    img = image.load_img(face_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)  # or version=2
    preds = vggface.predict(x)
    name, prob = utils.decode_predictions(preds)[0][0]
    if prob > 0.5:
        counter.update([name])
    else:
        counter.update(['unknown'])


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    face_dir = args['face_dir']
    vggface = VGGFace(model='resnet50')

    client = MongoClient(host=args['host'], port=27017)
    db = client.nytimes

    article_cursor = db.articles.find(
        {}, no_cursor_timeout=True).batch_size(128)

    counter = Counter()

    for i, article in tqdm(enumerate(article_cursor)):
        sections = article['parsed_section']
        image_positions = article['image_positions']
        for pos in image_positions:
            section = sections[pos]
            if 'facenet_details' not in section:
                continue

            face_path = os.path.join(
                face_dir, f"{section['hash']}_{pos:02}.jpg")
            classify_face(face_path, vggface, counter)

            # n_faces = section['facenet_details']['n_faces']
            # if n_faces > 1:
            #     for i in range(2, n_faces + 1):
            #         face_path = os.path.join(
            #             face_dir, f"{section['hash']}_{pos:02}_{i}.jpg")

        if i == 1000 or (i > 1000 and (i % 50000 == 0)):
            with open('data/nytimes/face_counter.pkl', 'wb') as f:
                pickle.dump(counter, f)


if __name__ == '__main__':
    main()
