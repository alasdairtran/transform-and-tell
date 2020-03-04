import copy
from typing import Dict, Iterable, List, Optional, Set, Union

from allennlp.data import Vocabulary
from allennlp.data import instance as adi  # pylint: disable=unused-import
from allennlp.data.vocabulary import (DEFAULT_NON_PADDED_NAMESPACES,
                                      _NamespaceDependentDefaultDict)
from transformers.tokenization_bert import load_vocab


class _RobertaTokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str,
                 oov_token: str) -> None:
        super().__init__(non_padded_namespaces,
                         lambda: {padding_token: 1,
                                  oov_token: 3},
                         lambda: {})


class _RobertaIndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str,
                 oov_token: str) -> None:
        super().__init__(non_padded_namespaces,
                         lambda: {1: padding_token,
                                  3: oov_token},
                         lambda: {})


@Vocabulary.register('roberta')
class RobertaVocabulary(Vocabulary):
    """Load vocabulary from a pre-trained Roberta vocab file.

    The __init__ is overwritten specifically so that we can match the
    unknown and pad token with those in the pretrained Roberta vocab.
    """

    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None,
                 min_pretrained_embeddings: Dict[str, int] = None,
                 padding_token: str = '<pad>',
                 oov_token: str = '<unk>') -> None:
        self._padding_token = padding_token
        self._oov_token = oov_token
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._token_to_index = _RobertaTokenToIndexDefaultDict(self._non_padded_namespaces,
                                                               self._padding_token,
                                                               self._oov_token)
        self._index_to_token = _RobertaIndexToTokenDefaultDict(self._non_padded_namespaces,
                                                               self._padding_token,
                                                               self._oov_token)
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        # Made an empty vocabulary, now extend it.
        self._extend(counter,
                     min_count,
                     max_vocab_size,
                     non_padded_namespaces,
                     pretrained_files,
                     only_include_pretrained_words,
                     tokens_to_add,
                     min_pretrained_embeddings)

    def _load_bert_vocab(self, vocab_path, namespace):
        vocab: Dict[str, int] = load_vocab(vocab_path)
        for word, idx in vocab.items():
            try:
                self._token_to_index[namespace][word] = idx
                self._index_to_token[namespace][idx] = word
            except:
                print(word, type(word), idx)
                raise

    def __setstate__(self, state):
        """
        Conversely, when we unpickle, we need to reload the plain dicts
        into our special DefaultDict subclasses.
        """
        # pylint: disable=attribute-defined-outside-init
        self.__dict__ = copy.copy(state)
        self._token_to_index = _RobertaTokenToIndexDefaultDict(self._non_padded_namespaces,
                                                               self._padding_token,
                                                               self._oov_token)
        self._token_to_index.update(state["_token_to_index"])
        self._index_to_token = _RobertaIndexToTokenDefaultDict(self._non_padded_namespaces,
                                                               self._padding_token,
                                                               self._oov_token)
        self._index_to_token.update(state["_index_to_token"])

        return vocab
