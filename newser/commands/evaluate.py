import json
import logging
import math
import os
import string
from typing import Any, Dict, Iterable

import spacy
import textstat
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
from nltk.tokenize import word_tokenize

from .train import yaml_to_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def evaluate_from_file(archive_path, model_path, overrides=None, eval_suffix='', device=0):
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
                       device, serialization_dir, eval_suffix, batch_weight_key='')

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = os.path.join(
        serialization_dir, f"evaluate-metrics{eval_suffix}.json")
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             serialization_dir: str,
             eval_suffix: str,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    nlp = spacy.load("en_core_web_lg")
    assert not os.path.exists(os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl'))
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

            write_to_json(output_dict, serialization_dir, nlp, eval_suffix)

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
            # if loss_count != batch_count:
            #     raise RuntimeError("The model you are trying to evaluate only sometimes " +
            #                        "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics


def write_to_json(output_dict, serialization_dir, nlp, eval_suffix):
    if 'captions' not in output_dict:
        return

    captions = output_dict['captions']
    generations = output_dict['generations']
    metadatas = output_dict['metadata']

    out_path = os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl')
    with open(out_path, 'a') as f:
        for i, caption in enumerate(captions):
            m = metadatas[i]
            generation = generations[i]
            caption_doc = nlp(m['caption'])
            gen_doc = nlp(generation)
            context_doc = nlp(m['context'])
            obj = {
                'caption': caption,
                'raw_caption': m['caption'],
                'generation': generation,
                'web_url': m['web_url'],
                'image_path': m['image_path'],
                'context': m['context'],
                'caption_names': get_proper_nouns(caption_doc),
                'generated_names': get_proper_nouns(gen_doc),
                'context_names': get_proper_nouns(context_doc),
                'caption_entities': get_entities(caption_doc),
                'generated_entities': get_entities(gen_doc),
                'context_entities': get_entities(context_doc),
                'caption_readability': get_readability_scores(m['caption']),
                'gen_readability': get_readability_scores(generation),
                'caption_np': get_narrative_productivity(m['caption']),
                'gen_np': get_narrative_productivity(generation),
            }

            if 'copied_texts' in output_dict:
                obj['copied_text'] = output_dict['copied_texts'][i]

            f.write(f'{json.dumps(obj)}\n')


def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities


def get_readability_scores(text):
    scores = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'text_standard': textstat.text_standard(text, float_output=True),
        'difficult_words': textstat.difficult_words(text) / len(text.split()),
    }
    return scores


def is_word(tok):
    return tok not in string.punctuation


def get_narrative_productivity(text):
    doc = word_tokenize(text)
    doc = list(filter(is_word, doc))
    n_words = len(doc)
    n_terms = len(set(doc))

    scores = {
        'basic_ttr': basic_ttr(n_terms, n_words),
        'root_ttr': root_ttr(n_terms, n_words),
        'corrected_ttr': corrected_ttr(n_terms, n_words),
        'herdan': herdan(n_terms, n_words),
        'summer': summer(n_terms, n_words),
        'maas': maas(n_terms, n_words),
    }

    return scores


def basic_ttr(n_terms, n_words):
    """ Type-token ratio (TTR) computed as t/w, where t is the number of unique
    terms/vocab, and w is the total number of words.
    (Chotlos 1944, Templin 1957)
    """
    if n_words == 0:
        return 0
    return n_terms / n_words


def root_ttr(n_terms, n_words):
    """ Root TTR (RTTR) computed as t/sqrt(w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        Also known as Guiraud's R and Guiraud's index.
        (Guiraud 1954, 1960)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(n_words)


def corrected_ttr(n_terms, n_words):
    """ Corrected TTR (CTTR) computed as t/sqrt(2 * w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        (Carrol 1964)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(2 * n_words)


def herdan(n_terms, n_words):
    """ Computed as log(t)/log(w), where t is the number of unique terms/vocab, and w is the
        total number of words.
        Also known as Herdan's C.
        (Herdan 1960, 1964)
    """
    if n_words <= 1:
        return 0
    return math.log(n_terms) / math.log(n_words)


def summer(n_terms, n_words):
    """ Computed as log(log(t)) / log(log(w)), where t is the number of unique terms/vocab, and
        w is the total number of words.
        (Summer 1966)
    """
    if n_words <= 1:
        return 0
    return math.log(math.log(n_terms)) / math.log(math.log(n_words))


def maas(n_terms, n_words):
    """ Maas's TTR, computed as (log(w) - log(t)) / (log(w) * log(w)), where t is the number of
        unique terms/vocab, and w is the total number of words. Unlike the other measures, lower
        maas measure indicates higher lexical richness.
        (Maas 1972)
    """
    # We cap this score at 0.2
    if n_words <= 1:
        return 0.2
    score = (math.log(n_words) - math.log(n_terms)) / \
        (math.log(n_words) ** 2)
    return min(score, 0.2)
