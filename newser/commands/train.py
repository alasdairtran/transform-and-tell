

import os

import yaml
from allennlp.commands.train import train_model
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params, parse_overrides, with_fallback
from allennlp.models import Model


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          cache_directory: str = None,
                          cache_prefix: str = None) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    We overwrite the AllenNLP function to support YAML config files. We also
    set the default serialization directory to be where the config file lives.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    cache_directory : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    cache_prefix : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    if parameter_filename.endswith(('.yaml', '.yml')):
        params = yaml_to_params(parameter_filename, overrides)
    else:
        params = Params.from_file(parameter_filename, overrides)

    if not serialization_dir:
        config_dir = os.path.dirname(parameter_filename)
        serialization_dir = os.path.join(config_dir, 'serialization')

    return train_model(params,
                       serialization_dir,
                       file_friendly_logging,
                       recover,
                       force,
                       cache_directory, cache_prefix)


def yaml_to_params(params_file: str, overrides: str = "") -> Params:
    # redirect to cache, if necessary
    params_file = cached_path(params_file)

    with open(params_file) as f:
        file_dict = yaml.safe_load(f)

    overrides_dict = parse_overrides(overrides)
    param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

    return Params(param_dict)
