from typing import Dict, List

import torch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util
from overrides import overrides

TokenList = List[TokenType]  # pylint: disable=invalid-name


class RareTextField(TextField):
    def __init__(self,
                 tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 context_tokens: List[Token],
                 most_common: Dict) -> None:
        super().__init__(tokens, token_indexers)
        self.context_tokens = context_tokens
        self.most_common = most_common

    @overrides
    def index(self, vocab: Vocabulary):
        token_arrays: Dict[str, TokenList] = {}
        indexer_name_to_indexed_token: Dict[str, List[str]] = {}
        token_index_to_indexer_name: Dict[str, str] = {}
        for indexer_name, indexer in self._token_indexers.items():
            token_indices = indexer.tokens_to_indices(
                self.tokens, vocab, indexer_name, self.context_tokens, self.most_common)
            token_arrays.update(token_indices)
            indexer_name_to_indexed_token[indexer_name] = list(
                token_indices.keys())
            for token_index in token_indices:
                token_index_to_indexer_name[token_index] = indexer_name
        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.

        rare_list = [tensor['rare_tokens'] for tensor in tensor_list]
        for tensor in tensor_list:
            del tensor['rare_tokens']

        output = util.batch_tensor_dicts(tensor_list)
        output['rare_tokens'] = rare_list
        return output
