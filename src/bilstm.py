import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset as dset


class BiLSTM(nn.Module):
    def __init__(self, train_dataset, word_embed_dim, tag_embed_dim, hidden_dim, num_layers, bias, mlp1_dim, mlp2_dim, p_dropout, word_dropout):
        super(BiLSTM, self).__init__()
        
        self.unk = int(train_dataset.special_dict[dset.UNK])
        self.pad = int(train_dataset.special_dict[dset.PAD])
        self.y_pad = int(train_dataset.special_dict[dset.y_PAD])
        self.word_dropout = word_dropout

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

        self.mlp1 = nn.Linear(int(hidden_dim*2), mlp1_dim, bias=bias)
        self.mlp2 = nn.Linear(int(hidden_dim*2), mlp1_dim, bias=bias)

        self.mlp_out = nn.Linear(mlp1_dim, mlp2_dim, bias=bias)
#         self.mlp_out1 = nn.Linear(mlp1_dim, mlp2_dim, bias=bias)
#         self.mlp_out2 = nn.Linear(mlp1_dim, mlp2_dim, bias=bias)

    def forward(self, words, tags, lens, device, prints=False):
        max_len = words.shape[1]

        print('words', words.shape) if prints else None
        # [batch_size, max_sentence_len]

        print('tags', tags.shape) if prints else None
        # [batch_size, max_sentence_len]

        print('lens', len(lens)) if prints else None
        # [batch_size]

        # word dropout
        if self.training:
            mask = (torch.rand(words.shape) > self.word_dropout).long().to(device)
            inv_mask = (-mask + 1).float().to(device)
            
            words = words * mask + inv_mask * self.unk
            tags = tags * mask + inv_mask * self.unk
        
        words = self.word_embedding_layer(words.long())
        print('word_embeds', words.shape) if prints else None
        # [batch_size, max_sentence_len, word_embed_dim]

        tags = self.tag_embedding_layer(tags.long())
        print('tag_embeds', tags.shape) if prints else None
        # [batch_size, max_sentence_len, tag_embed_dim]

        x = torch.cat((words, tags), -1)
        print('cat', x.shape) if prints else None
        # [batch_size, max_sentence_len, word_embed_dim + tag_embed_dim]

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
#         print('pack_padded_sequence', x.shape) if prints else None
        # [batch_size, packed_size, word_embed_dim + tag_embed_dim]

        x, _ = self.lstm(x)
#         print('lstm', x.shape) if prints else None
        # [batch_size, seq_length, 2*hidden_dim]

        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=self.pad, total_length=max_len)
        print('pad_packed_sequence', x.shape) if prints else None

        # [batch_size, packed_size, 2*hidden_dim]

#         X1 = x.unsqueeze(1)
#         print('X1', X1.shape)
        
#         Y1 = x.unsqueeze(2)
        
#         print('Y1', Y1.shape)
#         X2 = X1.repeat(1, x.shape[1], 1, 1)
        
#         Y2 = Y1.repeat(1, 1, x.shape[1], 1)
        
#         print(X2.shape, X2.shape)
#         x = torch.cat([X2, Y2],-1)
        
#         x = torch.cat([x.unsqueeze(1).repeat(1, x.shape[1], 1, 1),
#                        x.unsqueeze(2).repeat(1, 1, x.shape[1], 1),
#                       ], -1)
#         print(Z.shape)
        
#         x1 = x.repeat(1, 2, 1)
#         print('x1', x1.shape) if prints else None
        
#         x2 = x.unsqueeze(2).repeat(1, 1, 2, 1).view(x.size(0), -1, x.size(2))
#         print('x2', x2.shape) if prints else None

#         x = torch.cat((x1, x2), dim=2).view(x.size(0),
#                                             x.size(1),
#                                             x.size(1),
#                                             int(x.size(2)*2))
#         print('all_word_combs', x.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, 4*hidden_dim]

        x1 = self.mlp1(x)
        print('mlp1', x1.shape) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp1_dim]

#         x1 = F.tanh(x1)
#         print('tanh_x1', x1.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, mlp1_dim]

        x2 = self.mlp2(x)
        print('mlp2', x2.shape) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp1_dim]

#         x2 = F.tanh(x2)
#         print('tanh_x2', x2.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, mlp1_dim]

#         x1 = self.mlp_out1(x1)
#         print('mlp_out1', x1.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

#         x2 = self.mlp_out2(x2)
#         print('mlp_out2', x2.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

#         x = torch.einsum("abc,adc->abdc", x1, x2)
#         print('einsum', x.shape) if prints else None
#         # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

        x1 = x1.unsqueeze(1)
        print('unsqueeze_x1', x1.shape) if prints else None
#         print('x1', x1) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

        x2 = x2.unsqueeze(2)
        print('unsqueeze_x2', x2.shape) if prints else None
#         print('x2', x2) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

        x = x1 + x2
        print('outer_add_x1_x2', x.shape) if prints else None
#         print('x1 + x2', x) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

        x = F.tanh(x)
        print('tanh', x.shape) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp1_dim]

        x = self.mlp_out(x)
        print('mlp_out', x.shape) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]
    
        x = F.log_softmax(x, dim=2)
        print('log_softmax', x.shape) if prints else None
        # [batch_size, max_sentence_len, max_sentence_len, mlp2_dim]

        x = x.squeeze(-1)
        print('squeeze', x.shape) if prints else None
        # [batch_size, mlp2_dim, max_sentence_len] ????????????

        return x
    
    
    