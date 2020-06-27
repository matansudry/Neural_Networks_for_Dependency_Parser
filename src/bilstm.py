import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset as dset
from . import utils


class BiLSTM(nn.Module):
    def __init__(self,
                 train_dataset,
                 word_embed_dim=100,
                 tag_embed_dim=25,
                 hidden_dim=125,
                 num_layers=2,
                 bias=True,
                 attention_dim=100,
                 p_dropout=0.1):
        super(BiLSTM, self).__init__()

        self.pad = int(train_dataset.special_dict[dset.PAD])
        self.y_pad = int(train_dataset.special_dict[dset.y_PAD])

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

        self.attention = utils.AdditiveAttention(input_dim=int(hidden_dim*2),
                                                 attention_dim=attention_dim,
                                                 activation=nn.Tanh(),
                                                 bias=bias,
                                                 include_root=False)

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
        print('cat', x.shape) if prints else None  # [batch_size, max_len, word_embed_dim + tag_embed_dim]

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=self.pad, total_length=max_len)
        print('lstm_out', x.shape) if prints else None  # [batch_size, max_len, 2*hidden_dim]

        x = self.attention(x, prints=prints)
        print('attention', x.shape) if prints else None  # [batch_size, max_len, attention_dim]

        # log-softmax head dimention (dim=2)
        x = F.log_softmax(x, dim=2)
        print('log_softmax', x.shape) if prints else None   # [batch_size, max_len - 1, max_len, 1]

        return x
    
    
    