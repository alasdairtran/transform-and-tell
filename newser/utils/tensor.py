def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)
