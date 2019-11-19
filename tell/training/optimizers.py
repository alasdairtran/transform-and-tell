from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer
from pytorch_transformers.optimization import AdamW

Registrable._registry[Optimizer]['adam_w'] = AdamW

# Apex may not be available, because it has to be installed manually.
try:
    from apex.optimizers import FusedAdam
    Registrable._registry[Optimizer]['fused_adam'] = FusedAdam  # pylint: disable=protected-access
except ImportError:
    pass
