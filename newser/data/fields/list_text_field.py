from typing import Dict

from allennlp.data.fields import ListField
from overrides import overrides


class ListTextField(ListField):
    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super().get_padding_lengths()
        padding_lengths['total_num_tokens'] = padding_lengths['num_fields'] * \
            padding_lengths['list_num_tokens']
        return padding_lengths
