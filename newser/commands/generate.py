import os

import torch
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device
from allennlp.training.util import datasets_from_params

from .train import yaml_to_params


def generate(archive_path, model_path, overrides=None, device=0):
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
    if os.path.exists(os.path.join(serialization_dir, "vocabulary")):
        vocab_path = os.path.join(serialization_dir, "vocabulary")
        vocab = Vocabulary.from_files(vocab_path)

    if archive_path.endswith('yaml'):
        model = Model.from_params(vocab=vocab, params=config.pop('model'))

    if model_path:
        best_model_state = torch.load(model_path)
        model.load_state_dict(best_model_state)

    instances = all_datasets.get('validation')
    data_iterator = DataIterator.from_params(
        config.pop("validation_iterator"))
    data_iterator._batch_size = 1

    data_iterator.index_with(model.vocab)
    model.eval().to(device)

    with torch.no_grad():
        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
        generator_tqdm = Tqdm.tqdm(
            iterator, total=data_iterator.get_num_batches(instances))

        for batch in generator_tqdm:
            batch = move_to_device(batch, device)
            output_dict = model.generate(**batch)
            generated_text = output_dict['generated_text']
            print(generated_text)
            print(output_dict['caption'])
            print()
