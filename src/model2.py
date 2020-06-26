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
                 activation=nn.Tanh(),
                 p_dropout=0.1,
                 word_dropout=0.25,
                 attention=utils.AdditiveAttention,
                 softmax=nn.LogSoftmax(dim=2),
                 glove=True,
                 positional_encoding=False
                ):
        
        super(Model2, self).__init__()
        
        self.unk = int(train_dataset.special_dict[dset.UNK])
        self.pad = int(train_dataset.special_dict[dset.PAD])
        self.y_pad = int(train_dataset.special_dict[dset.y_PAD])
        self.word_dropout = word_dropout

        if glove:
            self.word_embedding_layer = None
        else:
            self.word_embedding_layer = nn.Embedding(num_embeddings=train_dataset.words_num,
                                                     embedding_dim=word_embed_dim,
                                                     padding_idx=self.pad)
        
        self.tag_embedding_layer = nn.Embedding(num_embeddings=train_dataset.tags_num,
                                                embedding_dim=tag_embed_dim,
                                                padding_idx=self.pad)

        # TODO: positional
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(int(word_embed_dim + tag_embed_dim),
                                                          dropout=p_dropout)
        else:
            self.positional_encoding = None
        
        # TODO: TransformerEncoder
        self.lstm = nn.LSTM(input_size=word_embed_dim + tag_embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bias=bias,
                            dropout=p_dropout,
                            batch_first=True,
                            bidirectional=True)

#         encoder_layers = nn.TransformerEncoderLayer(word_embed_dim + tag_embed_dim,
#                                                     nhead=num_layers,
#                                                     dim_feedforward=hidden_dim,
#                                                     dropout=p_dropout,
#                                                     activation='relu')
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        
#         self.mlp11 = nn.Linear(word_embed_dim + tag_embed_dim, mlp1_dim, bias=bias)
#         self.mlp12 = nn.Linear(word_embed_dim + tag_embed_dim, mlp1_dim, bias=bias)

        self.attention = attention(input_dim=int(hidden_dim*2),
                                   attention_dim=attention_dim,
                                   activation=activation,
                                   bias=bias,
                                   include_root=False)

#         self.mlp11 = nn.Linear(int(hidden_dim*2), mlp1_dim, bias=bias)
#         self.mlp12 = nn.Linear(int(hidden_dim*2), mlp1_dim, bias=bias)
        
#         self.activation = activation
        
#         self.mlp2 = nn.Linear(mlp1_dim, 1, bias=bias)
        
        self.softmax = softmax

    def forward(self, words, tags, lens, device, prints=False):
        max_len = words.shape[1]

        print('words', words.shape) if prints else None
        # [batch_size, max_len]

        print('tags', tags.shape) if prints else None
        # [batch_size, max_len]

        print('lens', len(lens)) if prints else None
        # [batch_size]

        # word dropout
        if self.training:
            mask = (torch.rand(tags.shape) > self.word_dropout).long().to(device)
            print('mask', mask.shape) if prints else None
            inv_mask = (-mask + 1).float().to(device)
            
            tags = tags * mask + inv_mask * self.unk
            
            if self.word_embedding_layer is not None:
                words = words * mask + inv_mask * self.unk
            else:
                mask = mask.unsqueeze(-1).repeat(1, 1, words.shape[-1])
                print('extended_mask', mask.shape) if prints else None
                words = words * mask

        if self.word_embedding_layer is not None:
            words = self.word_embedding_layer(words.long())
            print('word_embeds', words.shape) if prints else None
            # [batch_size, max_len, word_embed_dim]
    
        tags = self.tag_embedding_layer(tags.long())
        print('tag_embeds', tags.shape) if prints else None
        # [batch_size, max_len, tag_embed_dim]

        x = torch.cat((words, tags), -1)
        print('cat', x.shape) if prints else None
        # [batch_size, max_len, word_embed_dim + tag_embed_dim]

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
            print('positional_encoding', x.shape) if prints else None
            # [batch_size, max_len, word_embed_dim + tag_embed_dim]
        
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        # [batch_size, packed_size, word_embed_dim + tag_embed_dim]

#         x = self.transformer_encoder(x)
#         print('transformer_encoder', x.shape) if prints else None
        
        x, _ = self.lstm(x)
#         print('lstm', x.shape) if prints else None
        # [batch_size, packed_size, 2*hidden_dim]

        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=self.pad, total_length=max_len)
        print('pad_packed_sequence', x.shape) if prints else None
        # [batch_size, max_len, 2*hidden_dim]

        x = self.attention(x)
        
#         x1 = self.mlp11(x)
#         print('mlp11', x1.shape) if prints else None
#         # [batch_size, max_len, attention_dim]

#         x2 = self.mlp12(x)
#         print('mlp12', x2.shape) if prints else None
#         # [batch_size, max_len, attention_dim]

#         x1 = x1.unsqueeze(1)
#         print('unsqueeze_x1', x1.shape) if prints else None
#         # [batch_size, 1, max_len, attention_dim]

#         x2 = x2.unsqueeze(2)
#         print('unsqueeze_x2', x2.shape) if prints else None
#         # [batch_size, max_len, 1, mlp2_dim]

#         x = x1 + x2
#         print('outer_add_x1_x2', x.shape) if prints else None
#         # [batch_size, max_len, max_len, attention_dim]

#         # del ROOT from modifiers dimention (dim=1)
#         x = x[:, 1:, :, :]
#         print('del ROOT', x.shape) if prints else None
#         # [batch_size, max_len - 1, max_len, attention_dim]

#         x = self.activation(x)
#         print('activation', x.shape) if prints else None
#         # [batch_size, max_len - 1, max_len, attention_dim]

#         x = self.mlp2(x)
#         print('mlp2', x.shape) if prints else None
#         # [batch_size, max_len - 1, max_len, 1]
    
        # log-softmax head dimention (dim=2)
        x = self.softmax(x)
        print('softmax', x.shape) if prints else None
        # [batch_size, max_len - 1, max_len, 1]

#         x = x.squeeze(-1)
#         print('squeeze', x.shape) if prints else None
#         # [batch_size, max_len - 1, max_len]

        return x


