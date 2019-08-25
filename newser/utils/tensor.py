def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]
