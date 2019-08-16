from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sort_batch_by_length
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from pytorch_transformers.modeling_roberta import RobertaModel
from pytorch_transformers.modeling_utils import SequenceSummary

from .resnet import resnext101_32x16d_wsl


@dataclass
class SequenceSummaryConfig:
    hidden_size: int
    summary_type: str = 'last'
    summary_use_proj: bool = True
    summary_proj_to_labels: bool = False
    summary_activation: str = 'tanh'
    summary_first_dropout: float = 0.1
    summary_last_dropout: float = 0.1


class Attention(nn.Module):
    """Attention Network.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    """

    def __init__(self, n_channels, hidden_size, attention_dim):
        """
        :param n_channels: feature size of encoded images
        :param hidden_size: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super().__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(n_channels, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image, decoder_hidden):
        # image.shape == [batch_size, n_pixels, n_channels]
        # decoder_hidden.shape == [batch_size, hidden_size]

        attn_1 = self.encoder_att(image)
        # attn_1 = [batch_size, n_pixels, attention_size]

        attn_2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        # attn_2 = [batch_size, 1, attention_size]

        attn = self.full_att(self.relu(attn_1 + attn_2)).squeeze(2)
        # attn.shape == [batch_size, num_pixels]

        alpha = self.softmax(attn)
        # alpha.shape == [batch_size, num_pixels]

        attended_image = (image * alpha.unsqueeze(2)).sum(dim=1)
        # attended_image.shape = [batch_size, n_channels]

        return attended_image, alpha


@Model.register("baseline")
class BaselineModel(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 attention_dim: int = 512,
                 hidden_size: int = 512,
                 dropout: float = 0.1,
                 vocab_size: int = 50264,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 index: str = 'roberta',
                 padding_value: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        n_channels = 2048
        text_embed_size = 1024
        self.index = index
        self.namespace = namespace
        self.resnet = resnext101_32x16d_wsl()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.attention = Attention(n_channels, hidden_size, attention_dim)

        # Projection so that image and text embeds have same dimension
        # self.image_proj = nn.Linear(2048, 384)
        # self.text_proj = nn.Linear(1024, 384)

        # Linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(n_channels, hidden_size)

        # Linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(n_channels, hidden_size)

        # Linear layer to create a sigmoid-activation gate
        self.f_beta = nn.Linear(hidden_size, n_channels)

        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

        input_size = n_channels + text_embed_size
        self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        self.n_batches = 0
        self.n_samples = 0

        initializer(self)

        # Initialize the weight with first layer of BERT
        self.fc.weight.data.copy_(
            self.roberta.model.decoder.sentence_encoder.embed_tokens.weight)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        # STEP 1: Embed the caption
        # Sort the caption by decreasing lengths (save computation later on)
        caption_ids = caption[self.index]
        # caption_ids.shape == [batch_size, seq_len]

        # Assume the padding token ID is 1
        caption_mask = caption_ids != 1
        # caption_ids.shape == [batch_size, seq_len]

        caption_lens = caption_mask.sum(dim=1)
        # caption_len.shape == [batch_size]

        caption_ids, caption_lens, _, sort_index = sort_batch_by_length(
            caption_ids, caption_lens)

        caption_embeds = self.roberta.extract_features(
            caption_ids, return_all_hiddens=True)[0]
        # caption_embeds.shape == [batch_size, seq_len, 1024]

        # STEP 2: Embed the image
        # image.shape == [batch_size, 3, 224, 224]
        image = self.resnet(image)
        # image.shape == [batch_size, 2048, 7, 7]

        image_embeds = image.permute(0, 2, 3, 1)
        # image_embeds.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = image_embeds.shape
        P = H * W  # number of pixels
        image_embeds = image_embeds.view(B, P, C)
        image_embeds = image_embeds[sort_index]
        # image_embeds.shape == [batch_size, 49, 2048]

        # image_embeds = self.image_proj(image_embeds)
        # image_embeds.shape == [batch_size, 49, 1024]

        # STEP 3: Embed the first 512 words of the context
        # context_ids = context[self.index][:, :512]
        # # caption_ids.shape == [batch_size, seq_len]

        # context_embeds = self.roberta.extract_features(context_ids)
        # context_embeds = context_embeds[sort_index]

        # Initialize LSTM state
        h, c = self.init_hidden_state(image_embeds)

        # We won't decode at the </s> position, since we've finished
        # generating as soon as we generate </s>.
        decode_lens = (caption_lens - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        V = self.vocab.get_vocab_size(self.namespace)
        S = max(decode_lens)
        logits = image_embeds.new_zeros(B, S, V)
        alphas = image_embeds.new_zeros(B, S, P)

        # At each time-step, decode by attention-weighing the encoder's output
        # based on the decoder's previous hidden state output then generate a
        # new word in the decoder with the previous word and the attention
        # weighted encoding
        for t in range(S):
            batch_size_t = sum([length > t for length in decode_lens])
            image_embeds_t = image_embeds[:batch_size_t]
            caption_embeds_t = caption_embeds[:batch_size_t]
            h_t = h[:batch_size_t]
            c_t = c[:batch_size_t]

            attended_image, alpha = self.attention(image_embeds_t, h_t)
            # attended_image.shape = [batch_size_t, n_channels]
            # alpha.shape == [batch_size_t, num_pixels]

            gate = torch.sigmoid(self.f_beta(h_t))
            # gate.shape == [batch_size_t, n_channels]

            attended_image = gate * attended_image
            # attended_image.shape = [batch_size_t, n_channels]

            # The input to the RNN cell is a concatenation of
            #   1) The word from the previous step of the caption
            #   2) The attended image
            prev_word = caption_embeds_t[:, t, :]
            rnn_input = torch.cat([prev_word, attended_image], dim=1)

            # Feed through the RNN cell
            h, c = self.rnn_cell(rnn_input, (h_t, c_t))

            # Project onto the vocabulary
            logits_t = self.fc(self.dropout(h))
            # logits_t.shape = [batch_size_t, vocab_size]

            logits[:batch_size_t, t, :] = logits_t
            alphas[:batch_size_t, t, :] = alpha

        # Calculate loss
        probs = F.log_softmax(logits, dim=-1)
        # probs.shape == [batch_size, seq_len, vocab_size]

        probs = probs.view(-1, V)
        targets = caption_ids[:, 1:]

        padding_idx = 1
        text_mask = targets.ne(padding_idx)
        targets = targets[text_mask].contiguous().view(-1)
        probs = probs[text_mask.view(-1)]

        loss = F.nll_loss(probs, targets, reduction='mean')

        self.n_batches += 1
        self.n_samples += B

        output_dict = {'loss': loss}
        return output_dict

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        # STEP 1: Embed the caption
        # Sort the caption by decreasing lengths (save computation later on)
        caption_ids = caption[self.index]
        # caption_ids.shape == [batch_size, seq_len]

        # Assume the padding token ID is 1
        caption_mask = caption_ids != 1
        # caption_ids.shape == [batch_size, seq_len]

        caption_lens = caption_mask.sum(dim=1)
        # caption_len.shape == [batch_size]

        caption_ids, caption_lens, _, sort_index = sort_batch_by_length(
            caption_ids, caption_lens)

        caption_embeds = self.roberta.extract_features(
            caption_ids, return_all_hiddens=True)[0]
        # caption_embeds.shape == [batch_size, seq_len, 1024]

        # STEP 2: Embed the image
        # image.shape == [batch_size, 3, 224, 224]
        image = self.resnet(image)
        # image.shape == [batch_size, 2048, 7, 7]

        image_embeds = image.permute(0, 2, 3, 1)
        # image_embeds.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = image_embeds.shape
        P = H * W  # number of pixels
        image_embeds = image_embeds.view(B, P, C)
        image_embeds = image_embeds[sort_index]
        # image_embeds.shape == [batch_size, 49, 2048]

        # image_embeds = self.image_proj(image_embeds)
        # image_embeds.shape == [batch_size, 49, 1024]

        # STEP 3: Embed the first 512 words of the context
        # context_ids = context[self.index][:, :512]
        # # caption_ids.shape == [batch_size, seq_len]

        # context_embeds = self.roberta.extract_features(context_ids)
        # context_embeds = context_embeds[sort_index]

        # Initialize LSTM state
        h, c = self.init_hidden_state(image_embeds)

        # We won't decode at the </s> position, since we've finished
        # generating as soon as we generate </s>.
        decode_lens = (caption_lens - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        V = self.vocab.get_vocab_size(self.namespace)
        generated = caption_ids.new_zeros(B, 100)

        # At each time-step, decode by attention-weighing the encoder's output
        # based on the decoder's previous hidden state output then generate a
        # new word in the decoder with the previous word and the attention
        # weighted encoding.
        for t in range(100):
            attended_image, alpha = self.attention(image_embeds, h)
            # attended_image.shape = [batch_size_t, n_channels]
            # alpha.shape == [batch_size_t, num_pixels]

            gate = torch.sigmoid(self.f_beta(h))
            # gate.shape == [batch_size_t, n_channels]

            attended_image = gate * attended_image
            # attended_image.shape = [batch_size_t, n_channels]

            # The input to the RNN cell is a concatenation of
            #   1) The word from the previous step of the caption
            #   2) The attended image
            if t == 0:
                prev_word = caption_embeds[:, 0, :]
                generated[:, 0] = caption_ids[:, 0]
            else:
                # top-20 sampling
                probs_t = F.softmax(logits_t, dim=-1)
                # probs_t.shape == [batch_size_t, vocab_size]

                top_probs, top_indices = probs_t.topk(10, dim=-1)
                # top_indices.shape == [batch_size_t, 10]

                idx = torch.multinomial(top_probs, 1)
                # idx.shape == [batch_size, 1]

                word_idx = torch.gather(top_indices, dim=1, index=idx)
                word_idx = word_idx.squeeze(-1)
                # word_idx.shape == [batch_size]

                generated[:, t] = word_idx

                if word_idx.item() == 2:
                    break

                prev_word = self.roberta.extract_features(
                    generated[:, :t+1], return_all_hiddens=True)
                prev_word = prev_word[0][:, -1]
                # prev_word.shape == [batch_size, embed_size]

            rnn_input = torch.cat([prev_word, attended_image], dim=1)

            # Feed through the RNN cell
            h, c = self.rnn_cell(rnn_input, (h, c))

            # Project onto the vocabulary
            logits_t = self.fc(self.dropout(h))
            # logits_t.shape = [batch_size_t, vocab_size]

        self.n_batches += 1
        self.n_samples += B

        generated_indices = generated[0, :t].cpu()
        generated_text = self.roberta.decode(generated_indices)

        return {
            'generated_indices': generated_indices,
            'generated_text': generated_text,
            'caption': metadata[0]['caption'],
        }

    def init_hidden_state(self, image_embeds):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # image_embeds.shape == [batch_size, n_pixels, n_channels]
        mean_image_embeds = image_embeds.mean(dim=1)
        # mean_image_embeds.shape == [batch_size, n_pixels, n_channels]

        h = self.init_h(mean_image_embeds)  # (batch_size, decoder_dim)
        c = self.init_c(mean_image_embeds)
        return h, c

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics['_n_batches'] = self.n_batches
        metrics['_n_samples'] = self.n_samples

        if reset:
            self.n_batches = 0
            self.n_samples = 0

        return metrics
