"""Get articles from the New York Times API.

Usage:
    process_images.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -i --in-dir DIR   Root directory of data [default: data/goodnews/images].
    -i --out-dir DIR   Root directory of data [default: data/goodnews/images_processed].

"""
import os
from glob import glob

import ptvsd
import torchvision.transforms.functional as F
from docopt import docopt
from PIL import Image
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.utils import setup_logger

logger = setup_logger()


def process_images(in_dir, out_dir):
    image_paths = glob(f'{in_dir}/*.jpg')
    for path in tqdm(image_paths):
        image_name = os.path.basename(path)
        out_path = os.path.join(out_dir, image_name)
        if os.path.exists(out_path):
            continue

        try:
            with Image.open(path) as image:
                image = image.convert('RGB')
                image = F.resize(image, 256, Image.ANTIALIAS)
                image = F.center_crop(image, (224, 224))
                image.save(out_path, image.format)
        except OSError:
            continue

def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'in_dir': os.path.exists,
        'out_dir': str,
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

    os.makedirs(args['out_dir'], exist_ok=True)

    process_images(args['in_dir'], args['out_dir'])


if __name__ == '__main__':
    main()
