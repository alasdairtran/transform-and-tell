from allennlp.common.registrable import Registrable
from torch.nn.modules.loss import _Loss


class Criterion(_Loss, Registrable):
    pass
