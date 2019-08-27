import torch.nn.functional as F
import torch


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)
