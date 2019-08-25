from torch.nn.modules.loss import _Loss
from allennlp.common.registrable import Registrable


class Criterion(_Loss, Registrable):
    pass
