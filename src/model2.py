import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset as dset
from . import utils


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model2(nn.Module):
    def __init__(self,
                 train_dataset,
                 word_embed_dim=300,
                 tag_embed_dim=25,
                 hidden_dim=125,
                 num_layers=2,
                 bias=True,
                 attention_dim=100,
                 lstm_activation=None,
                 attn_activation=nn.Tanh(),
                 p_dropout=0.1,
                 attention=utils.AdditiveAttention,
                 softmax=nn.LogSoftmax(dim=2),
                 freeze=True,
                 glove=True):
        super(Model2, self).__init__()
        
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


