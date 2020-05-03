"""Server

Run NLP Server.

Usage:
    server.py [options] TASK

Options:
    -h --help           Show this screen.
    --version           Show version.
    -u --pudb           Enable debug mode with pudb.
    -p --ptvsd PORT     Enable debug mode with ptvsd on a given port, for
                        example 5678.
    -n --n-workers INT  Number of workers [default: 1].
    --port INT          Port in [default: 5558].
    --port-out INT      Port out [default: 5559].
    TASK                One of: coref, grid.
"""
import ptvsd
import pudb
from docopt import docopt
from schema import And, Or, Schema, Use

from .base import NLPServer


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'n_workers': Use(int),
        'port': Use(int),
        'port_out': Use(int),
        object: object,
    })
    args = schema.validate(args)
    args['debug'] = args['ptvsd'] or args['pudb']
    return args


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)
    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()
    elif args['pudb']:
        pudb.set_trace()

    with NLPServer(task=args['task'],
                   n_workers=args['n_workers'],
                   port=args['port'],
                   port_out=args['port_out']) as server:
        server.join()


if __name__ == '__main__':
    main()
