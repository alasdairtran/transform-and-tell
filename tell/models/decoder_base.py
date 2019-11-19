import torch.nn as nn
from allennlp.common.registrable import Registrable


class Decoder(Registrable, nn.Module):
    pass


class DecoderLayer(Registrable, nn.Module):
    pass
