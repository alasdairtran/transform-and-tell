"""Get articles from the New York Times API.

Usage:
    get_nytimes.py [options] API_KEY

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""

import json
import os
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import ptvsd
from docopt import docopt
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
        object: object,
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

    api_key = args['api_key']
    url = 'http://api.nytimes.com/svc/archive/v1/%s/%s.json?api-key=%s'

    data_dir = 'data/nytimes/archive'
    os.makedirs(data_dir, exist_ok=True)

    for year, month in tqdm(month_year_iter(8, 2019, 12, 1979)):
        out_path = f'{data_dir}/{year}_{month:02}.json'
        if os.path.exists(out_path):
            continue

        start = time.time()
        request_string = url % (year, month, api_key)

        while True:
            try:
                response = urlopen(request_string)
            except HTTPError:
                time.sleep(10)
                continue
            break

        raw_content = response.read()
        content = json.loads(raw_content)
        with open(out_path, 'w') as f:
            json.dump(content, f)

        # Note that there's a limit of 4,000 requests per day and 10
        # requests per minute. We should sleep 6 seconds between calls to
        # avoid hitting the per minute rate limit.
        elapsed = time.time() - start
        time.sleep(max(0, 6 - elapsed))


if __name__ == '__main__':
    main()
