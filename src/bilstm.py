import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset as dset
from . import utils


class BiLSTM(nn.Module):
    """
    an implementation of the https://arxiv.org/abs/1603.04351 paper by Eliyahu Kiperwasser, Yoav Goldberg:
    "Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations"
    with various tuneable hyperparameters for possible improvements.
    """
    def __init__(self,
                 train_dataset,
                 word_embed_dim=300,
                 tag_embed_dim=25,
                 hidden_dim=125,
                 num_layers=2,
                 bias=True,
                 lstm_activation=None,
                 p_dropout=0.1,
                 attention=utils.AdditiveAttention,
                 attention_dim=100,
                 attn_activation=nn.Tanh(),
                 softmax=nn.LogSoftmax(dim=2),
                 glove=False,
                 freeze=False):
        super(BiLSTM, self).__init__()
        """
        Parameters
        -------
        train_dataset : an instance of dataset.DataSet
            the train dataset
        word_embed_dim : int
            word embedding dimention (default is 300)
        tag_embed_dim : int
            POS tag embedding dimention (default is 25)
        hidden_dim : int
            LSTM hidden_size (default is 125)
        num_layers : int
            LSTM num_layers (default is 2)
        bias : bool
            LSTM and Linear (Attention) modules bias (default is True)
        lstm_activation : nn.Module or None, optional
            optional activation layer after LSTM (default is None)
        p_dropout : float between 0 and 1
            LSTM dropout rate (default is 0.1)
        attention : utils.AdditiveAttention or utils.DotAttention or utils.MultiplicativeAttention
            attention layer for calculating dependency parsing head probabilities (default is utils.AdditiveAttention)
        attn_activation : nn.Module, optional
            attention layer activation, only used for utils.AdditiveAttention (default is nn.Tanh())
        softmax : mm.Module
            path to be used for versions dirs (default is nn.LogSoftmax(dim=2))
        glove : bool
            use glove.6B.300d embedings (default is False)
        freeze : bool
            freeze glove weights, only used if glove=True (default is False)

        """
        self.pad = int(train_dataset.special_dict[dset.PAD])
        self.y_pad = int(train_dataset.special_dict[dset.y_PAD])

        if glove:
            assert train_dataset.words_vectors is not None, 'train_dataset.words_vectors must be glove embedings'
            assert word_embed_dim == train_dataset.words_vectors.shape[1], \
                f'word_embed_dim={word_embed_dim} != train_dataset.words_vectors.shape[1]={train_dataset.words_vectors.shape[1]}'
            self.word_embedding_layer = nn.Embedding.from_pretrained(train_dataset.words_vectors, freeze=freeze)
        else:
            self.word_embedding_layer = nn.Embedding(num_embeddings=train_dataset.words_num,
                                                     embedding_dim=word_embed_dim,
                                                     padding_idx=self.pad)

        self.tag_embedding_layer = nn.Embedding(num_embeddings=train_dataset.tags_num,
                                                embedding_dim=tag_embed_dim,
                                                padding_idx=self.pad)

        self.lstm = nn.LSTM(input_size=word_embed_dim + tag_embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bias=bias,
                            dropout=p_dropout,
                            batch_first=True,
                            bidirectional=True)

        self.lstm_activation = lstm_activation

        self.attention = attention(input_dim=int(hidden_dim*2),
                                   attention_dim=attention_dim,
                                   activation=attn_activation,
                                   bias=bias,
                                   include_root=False)

        self.softmax = softmax

    def forward(self, words, tags, lens, device, prints=False):
        max_len = words.shape[1]

        print('words', words.shape) if prints else None  # [batch_size, max_len]
        print('tags', tags.shape) if prints else None  # [batch_size, max_len]
        print('lens', len(lens)) if prints else None  # [batch_size]

        words = self.word_embedding_layer(words.long())
        print('word_embeds', words.shape) if prints else None  # [batch_size, max_len, word_embed_dim]

        tags = self.tag_embedding_layer(tags.long())
        print('tag_embeds', tags.shape) if prints else None  # [batch_size, max_len, tag_embed_dim]

        x = torch.cat((words, tags), -1)
        print('cat words, tags', x.shape) if prints else None  # [batch_size, max_len, word_embed_dim + tag_embed_dim]

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=self.pad, total_length=max_len)
        print('lstm', x.shape) if prints else None  # [batch_size, max_len, 2*hidden_dim]

        if self.lstm_activation is not None:
            x = self.lstm_activation(x)
            print('lstm_activation', x.shape) if prints else None  # [batch_size, max_len, 2*hidden_dim]

        x = self.attention(x)
        print('attention', x.shape) if prints else None  # [batch_size, max_len - 1, max_len]

        # log-softmax head dimention (dim=2)
        x = self.softmax(x)
        print('softmax', x.shape) if prints else None  # [batch_size, max_len - 1, max_len]

        return x


