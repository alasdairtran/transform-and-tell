import re
from typing import Dict, List, Tuple

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

SPACE_NORMALIZER = re.compile(r"\s+")


def to_token_ids(sentence, roberta):
    bpe_tokens = roberta.bpe.encode(sentence)
    bpe_tokens = f'<s> {bpe_tokens} </s>'
    words = tokenize_line(bpe_tokens)

    token_ids = []
    for word in words:
        idx = roberta.task.source_dictionary.indices[word]
        token_ids.append(idx)
    return token_ids


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@TokenIndexer.register("roberta_names_matched")
class RobertaNamesMatchedTokenIndexer(TokenIndexer[int]):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 legacy: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 token_min_padding_length: int = 0,
                 padding_on_right: bool = True,
                 padding_value: int = 1,
                 max_len: int = 512) -> None:
        super().__init__(token_min_padding_length)
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.source_dictionary = roberta.task.source_dictionary
        self.bpe = roberta.bpe.bpe
        self.bpe_legacy = roberta.bpe
        self._added_to_vocabulary = False
        self._namespace = namespace
        self._padding_on_right = padding_on_right
        self._padding_value = padding_value
        self._max_len = max_len
        self.legacy = legacy

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
                          index_name: str,
                          copy_infos: Dict[str, List[Tuple[int, int]]] = None,
                          proper_infos=None,
                          key: str = None) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text = ' '.join([token.text for token in tokens])
        indices, copy_masks, proper_masks = self.encode(
            text, copy_infos, proper_infos, key)

        output = {
            index_name: indices,
            f'{index_name}_copy_masks': copy_masks,
        }

        if proper_masks is not None:
            output[f'{index_name}_proper_masks'] = proper_masks

        return output

    def encode(self, sentence, copy_infos, proper_infos, key):
        bpe_tokens, copy_masks, proper_masks = self._byte_pair_encode(
            sentence, copy_infos, proper_infos, key)
        sentence = ' '.join(map(str, bpe_tokens))
        words = tokenize_line(sentence)
        assert len(words) == len(copy_masks)

        # Enforce maximum length constraint
        words = words[:self._max_len - 2]
        copy_masks = copy_masks[:self._max_len - 2]
        words = ['<s>'] + words + ['</s>']
        copy_masks = [0] + copy_masks + [0]

        if proper_masks is not None:
            proper_masks = proper_masks[:self._max_len - 2]
            proper_masks = [0] + proper_masks + [0]

        token_ids = []
        for word in words:
            idx = self.source_dictionary.indices[word]
            token_ids.append(idx)

        return token_ids, copy_masks, proper_masks

    def _byte_pair_encode(self, text, copy_infos, proper_infos, key):
        bpe_tokens = []
        bpe_copy_masks = []

        raw_tokens = self.bpe.re.findall(self.bpe.pat, text)
        # e.g.[' Tomas', ' Maier', ',', ' autumn', '/', 'winter', ' 2014', ',', '\n', ' in', 'Milan', '.']

        copy_masks = self.get_copy_mask(raw_tokens, copy_infos, key)
        # Same length as raw_tokens

        if proper_infos is not None:
            proper_masks = self.get_copy_mask(
                raw_tokens, proper_infos, key)
        else:
            proper_masks = None

        for raw_token, copy_mask in zip(raw_tokens, copy_masks):
            # e.g. raw_token == " Tomas"

            # I guess this step is used so that we can distinguish between
            # the space separator and the space character.
            token = ''.join(self.bpe.byte_encoder[b]
                            for b in raw_token.encode('utf-8'))
            # e.g. token == "ĠTomas"

            token_ids = [self.bpe.encoder[bpe_token]
                         for bpe_token in self.bpe.bpe(token).split(' ')]
            # e.g. token_ids == [6669, 959]

            bpe_tokens.extend(token_ids)

            bpe_copy_masks.extend([copy_mask] * len(token_ids))

        if proper_masks is not None:
            bpe_proper_masks = []
            for raw_token, copy_mask in zip(raw_tokens, proper_masks):
                # e.g. raw_token == " Tomas"

                # I guess this step is used so that we can distinguish between
                # the space separator and the space character.
                token = ''.join(self.bpe.byte_encoder[b]
                                for b in raw_token.encode('utf-8'))
                # e.g. token == "ĠTomas"

                token_ids = [self.bpe.encoder[bpe_token]
                             for bpe_token in self.bpe.bpe(token).split(' ')]
                # e.g. token_ids == [6669, 959]

                bpe_proper_masks.extend([copy_mask] * len(token_ids))
        else:
            bpe_proper_masks = None

        return bpe_tokens, bpe_copy_masks, bpe_proper_masks

    def get_copy_mask(self, tokens, copy_infos, key):
        # We first compute the start and end points for each token.
        # End points are exclusive.
        # e.g. tokens = [' Tomas', ' Maier', ',', ' autumn', '/', 'winter', ' 2014', ',', '\n', ' in', 'Milan', '.']
        starts = []
        ends = []
        current = 0
        for token in tokens:
            starts.append(current)
            current += len(token)
            ends.append(current)

        copy_masks = [0] * len(tokens)

        if copy_infos is None and key is None:
            return copy_masks

        for idx, (name, info) in enumerate(copy_infos.items()):
            for c_start, c_end in info[key]:
                # A token is part of an entity if it lies strictly inside it
                for i, (start, end, token) in enumerate(zip(starts, ends, tokens)):
                    if token[0] == ' ':
                        c_start_adjusted = c_start - 1
                    else:
                        c_start_adjusted = c_start

                    if start >= c_start_adjusted and end <= c_end:
                        # if name not in token:
                        #     print(name, token)
                        # assert name in token
                        copy_masks[i] = idx + 1

        return copy_masks

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
            if 'copy_masks' in key or 'proper_masks' in key:
                def default_value(): return -1
            else:
                def default_value(): return self._padding_value
            padded_val = pad_sequence_to_length(sequence=val,
                                                desired_length=desired_num_tokens[key],
                                                default_value=default_value,
                                                padding_on_right=self._padding_on_right)
            padded_dict[key] = torch.LongTensor(padded_val)
        return padded_dict

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f'{index_name}_copy_masks']
