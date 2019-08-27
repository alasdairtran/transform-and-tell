from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from pytorch_transformers.tokenization_roberta import RobertaTokenizer


class RobertaTokenizer2(RobertaTokenizer):
    @overrides
    def convert_tokens_to_ids(self, tokens):
        """Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
        (resp. a sequence of ids), using the vocabulary.

        We manually cut off long sequences at 512 pieces.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return ids


@TokenIndexer.register("roberta")
class RobertaTokenIndexer(TokenIndexer[int]):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 token_min_padding_length: int = 0,
                 padding_on_right: bool = True,
                 padding_value: int = 1,
                 max_len: int = 512) -> None:
        super().__init__(token_min_padding_length)
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.source_dictionary = roberta.task.source_dictionary
        self.bpe = roberta.bpe
        self._added_to_vocabulary = False
        self._namespace = namespace
        self._padding_on_right = padding_on_right
        self._padding_value = padding_value
        self._max_len = max_len

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        for piece, idx in self.source_dictionary.indices.items():
            vocabulary._token_to_index[self._namespace][piece] = idx
            vocabulary._index_to_token[self._namespace][idx] = piece

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text = ' '.join([token.text for token in tokens])
        indices = self.encode(text)

        return {index_name: indices}

    def encode(self, sentence):
        bpe_sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        tokens = self.source_dictionary.encode_line(
            bpe_sentence, append_eos=False)
        return tokens.long().tolist()[:self._max_len]

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        padded_dict: Dict[str, torch.Tensor] = {}
        for key, val in tokens.items():
            padded_val = pad_sequence_to_length(sequence=val,
                                                desired_length=desired_num_tokens[key],
                                                default_value=lambda: self._padding_value,
                                                padding_on_right=self._padding_on_right)
            padded_dict[key] = torch.LongTensor(padded_val)
        return padded_dict

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name]
