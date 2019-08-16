"""Train and run semantic diff models.

Usage:
    newser (train|generate) [options] PARAM_PATH
    newser (-h | --help)
    newser (-v | --version)

Options:
    -e --expt-dir EXPT_PATH
                        Directory to store experiment results and model files.
                        If not given, they will be stored in the same directory
                        as the parameter file.
    -r, --recover       Recover training from existing model.
    -f, --force    Delete existing models and logs.
    -o --overrides OVERRIDES
                        A JSON structure used to override the experiment
                        configuration.
    -u --pudb           Enable debug mode with pudb.
    -p --ptvsd PORT     Enable debug mode with ptvsd on a given port, for
                        example 5678.
    -g --file-friendly-logging
                        Outputs tqdm status on separate lines and slows tqdm
                        refresh rate
    -i --include-package PACKAGE
                        Additional packages to include.
    -q --quiet          Print less info
    PARAM_PATH          Path to file describing the model parameters.
    -m --model-path PATH Path the the best model.

Examples:
    newser train -r -g expt/writing-prompts/lstm/config.yaml
"""

import logging
import os

import ptvsd
import pudb
from docopt import docopt
from schema import And, Or, Schema, Use

from newser.utils import setup_logger

from .generate import generate
from .train import train_model_from_file

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'param_path': Or(None, os.path.exists),
        'model_path': Or(None, os.path.exists),
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        object: object,
    })
    args = schema.validate(args)
    args['debug'] = args['ptvsd'] or args['pudb']
    return args


def main():
    """Parse command line arguments and execute script."""
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['debug']:
        logger.setLevel(logging.DEBUG)
    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address, redirect_output=True)
        ptvsd.wait_for_attach()
    elif args['pudb']:
        pudb.set_trace()

    if args['quiet']:
        # Disable some of the more verbose logging statements
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger(
            'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    if args['train']:
        train_model_from_file(
            parameter_filename=args['param_path'],
            serialization_dir=args['expt_dir'],
            overrides=args['overrides'],
            file_friendly_logging=args['file_friendly_logging'],
            recover=args['recover'],
            force=args['force'])

    elif args['generate']:
        generate(args['param_path'], args['model_path'], args['overrides'])


if __name__ == '__main__':
    main()
