from typing import Dict

import numpy as np
import torch
from allennlp.data.fields import Field
from overrides import overrides
from PIL import Image
from torchvision.transforms import Compose


class ImageField(Field[np.array]):
    """
    An ``ImageField`` stores an image as a ``np.ndarray`` which must have exactly three
    dimensions.

    Adapted from https://github.com/sethah/allencv/blob/master/allencv/data/fields/image_field.py

    Parameters
    ----------
    image: ``np.ndarray``
    """

    def __init__(self,
                 image: Image,
                 preprocess: Compose,
                 padding_value: int = 0) -> None:

        self.image = preprocess(image)
        self.padding_value = padding_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'channels': self.image.shape[0],
            'height': self.image.shape[1],
            'width': self.image.shape[2],
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return self.image

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return ImageField(np.empty(self.image.shape), Compose([]),
                          padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"ImageField with shape: {self.image.shape}."
