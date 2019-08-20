import json
import logging
import os

import torch
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device
from allennlp.training.util import datasets_from_params, evaluate

from .train import yaml_to_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def evaluate_from_file(archive_path, model_path, overrides=None, device=0):
    if archive_path.endswith('gz'):
        archive = load_archive(archive_path, device, overrides)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        serialization_dir = os.path.dirname(archive_path)
    elif archive_path.endswith('yaml'):
        config = yaml_to_params(archive_path, overrides)
        prepare_environment(config)
        config_dir = os.path.dirname(archive_path)
        serialization_dir = os.path.join(config_dir, 'serialization')

    all_datasets = datasets_from_params(config)

    # We want to create the vocab from scratch since it might be of a
    # different type. Vocabulary.from_files will always create the base
    # Vocabulary instance.
    # if os.path.exists(os.path.join(serialization_dir, "vocabulary")):
    #     vocab_path = os.path.join(serialization_dir, "vocabulary")
    #     vocab = Vocabulary.from_files(vocab_path)

    vocab = Vocabulary.from_params(config.pop('vocabulary'))
    model = Model.from_params(vocab=vocab, params=config.pop('model'))

    if model_path:
        best_model_state = torch.load(model_path)
        model.load_state_dict(best_model_state)

    instances = all_datasets.get('validation')
    iterator = DataIterator.from_params(
        config.pop("validation_iterator"))
    iterator._batch_size = 1

    iterator.index_with(model.vocab)
    model.eval().to(device)

    metrics = evaluate(model, instances, iterator,
                       device, batch_weight_key='')

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = os.path.join(serialization_dir, "evaluate-metrics.json")
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics
