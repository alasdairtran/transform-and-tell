import json
import logging
import os
from typing import Any, Dict, Iterable

import spacy
import torch
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from allennlp.nn.util import move_to_device
from allennlp.training.util import HasBeenWarned, datasets_from_params

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

    instances = all_datasets.get('test')
    iterator = DataIterator.from_params(
        config.pop("validation_iterator"))

    iterator.index_with(model.vocab)
    model.eval().to(device)
    model.evaluate_mode = True

    metrics = evaluate(model, instances, iterator,
                       device, serialization_dir, batch_weight_key='')

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = os.path.join(serialization_dir, "evaluate-metrics.json")
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             serialization_dir: str,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(
            iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            write_to_json(output_dict, serialization_dir, nlp)

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics


def write_to_json(output_dict, serialization_dir, nlp):
    if 'captions' not in output_dict:
        return

    captions = output_dict['captions']
    generations = output_dict['generations']
    metadatas = output_dict['metadata']

    out_path = os.path.join(serialization_dir, 'generations.jsonl')
    with open(out_path, 'a') as f:
        for caption, generation, m in zip(captions, generations, metadatas):
            obj = {
                'caption': caption,
                'raw_caption': m['caption'],
                'generation': generation,
                'web_url': m['web_url'],
                'image_path': m['image_path'],
                'context': m['context'],
                'caption_names': get_proper_nouns(m['caption'], nlp),
                'generated_names': get_proper_nouns(generation, nlp),
            }
            f.write(f'{json.dumps(obj)}\n')


def get_proper_nouns(text, nlp):
    doc = nlp(text)
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns
