import os
import csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

seed = 42
np.random.seed(seed)


ROOT = '<root>'
UNK = '<unk>'
PAD = '<pad>'
y_PAD = '<y_pad>'
SPECIAL = [y_PAD, PAD, UNK, ROOT]

glove_dim = 300

class DataSet:
    """
    a general DataSet for dependecy parsing
    """
    def __init__(self,
                 path,
                 train_dataset=None,
                 tagged=True,
                 use_glove=False,
                 tqdm_bar=False
                ):
        """
        Parameters
        -------
        path : str
            the path to the csv
        train_dataset : a DataSet instance
            should contain the train_dataset for creating validation/test datasets,
            it is important for the validation/test dataset to have the same vocabulary and word indecies as the train dataset.
            (default is None)
        tag_embed_dim : int
            POS tag embedding dimention (default is 25)
        tagged : bool
            is the dataset tagged (default is True)
        use_glove : bool
            use the pretrained glove.6B.300d weights (default is False)
        tqdm_bar : bool
            show a progress bar for loading (default is False)
        """
        # load csv to pd.DataFrame
        self.df = pd.read_csv(path, names=['Token_Counter', 'Token', 3, 'Token_POS', 5 ,6 , 'Token_Head', 8, 9, 10],
                              sep='\t', skip_blank_lines=False)
        self.df_preds = None
        
        self.word_frequency = pd.DataFrame(self.df['Token'].dropna().value_counts())
        
        self.words_vectors = None
        if train_dataset is not None:
            if use_glove:
                assert train_dataset.words_vectors is not None, 'train_dataset.words_vectors must be glove embedings'
                self.words_vectors = train_dataset.words_vectors
            self.max_sentence_len = train_dataset.max_sentence_len

            self.words_dict = train_dataset.words_dict
            self.tags_dict = train_dataset.tags_dict
            self.special_dict = train_dataset.special_dict

            self.words_num = train_dataset.words_num
            self.tags_num = train_dataset.tags_num
        else:
            if use_glove:
                from collections import Counter
                from torchtext.vocab import Vocab

                glove = Vocab(Counter(dict(self.word_frequency['Token'])), vectors="glove.6B.300d", specials=SPECIAL)
                self.words_dict = glove.stoi
                self.words_vectors = glove.vectors
            else:
                self.words_dict = {token: i + len(SPECIAL) - 1 for i, token in enumerate(sorted(list(set(self.df['Token'].dropna().values))))}

            # self.max_sentence_len is used as self.X, self.y 2nd dimention
            self.max_sentence_len = self.df['Token_Counter'].dropna().max() + 1

            self.tags_dict = {token: i + len(SPECIAL) - 1 for i, token in enumerate(sorted(list(set(self.df['Token_POS'].dropna().values))))}
            self.special_dict = {token: i - 1 for i, token in enumerate(SPECIAL)}

            self.words_num = len(self.words_dict) + len(self.special_dict) - 1
            self.tags_num = len(self.tags_dict) + len(self.special_dict) - 1

        # fill X, y tensors with data
        self.words_tensor = []
        self.tags_tensor = []
        self.lens = []
        self.y = []

        if tqdm_bar:
            from tqdm import tqdm
            iterable = tqdm(self.df.iterrows(), total=len(self.df))
        else:
            iterable = self.df.iterrows()

        for _, line in iterable:
            # i is df['Token_Counter']
            i = line.values[0]
            if i == 1.0:  # if i==1: init the sentence X, y
                sentence_words = [self.special_dict[ROOT]]
                sentence_tags = [self.special_dict[ROOT]]
                sentence_y = []

            if pd.notna(i):
                sentence_words.append(self.words_dict.get(line['Token'], self.special_dict[UNK]))

                sentence_tags.append(self.tags_dict.get(line['Token_POS'], self.special_dict[UNK]))
                if tagged:
                    sentence_y.append(line['Token_Head'])
                else:
                    sentence_y.append(0.0)
            else:
                self.words_tensor.append(torch.LongTensor(sentence_words))

                self.tags_tensor.append(torch.LongTensor(sentence_tags))
                self.lens.append(len(sentence_words))
                self.y.append(torch.FloatTensor(sentence_y))

        self.words_tensor = nn.utils.rnn.pad_sequence(self.words_tensor,
                                                      batch_first=False,
                                                      padding_value=self.special_dict[PAD]).transpose(1, 0)

        self.tags_tensor = nn.utils.rnn.pad_sequence(self.tags_tensor, batch_first=False, padding_value=self.special_dict[PAD]).transpose(1, 0)
        self.y = nn.utils.rnn.pad_sequence(self.y, batch_first=False, padding_value=self.special_dict[y_PAD]).transpose(1, 0)

    def dataset(self, word_dropout_alpha=0.25, train=False):
        """
        Parameters
        -------
        word_dropout_alpha : float > 0
            alpha word dropout rate, if word_dropout_alpha > 0 the word will be dropped for the training session with p = alpha/(word_tf + alpha)
        train : bool
            if train=True, word dropout will be applied with alpha=word_dropout_alpha
        """
        if train and word_dropout_alpha > 0.0:
            temp = self.word_frequency.copy()

            temp['Token'] = word_dropout_alpha/(temp['Token'] + word_dropout_alpha)
            rnd = np.random.random(len(temp))
            temp['Token'] = (temp['Token'] > rnd)

            drop = temp[temp['Token']].index
            drop = [self.words_dict[token] for token in drop]

            mask = mask = -(torch.Tensor(np.isin(self.words_tensor, drop)) - 1)

            new_words_tensor = (mask * self.words_tensor).long()
        else:
            new_words_tensor = self.words_tensor

        return list(zip(new_words_tensor, self.tags_tensor, self.lens, self.y))

    def insert_predictions(self, preds, name, tqdm_bar=False):
        self.df_preds = self.df.copy()
        counter = 0
        if tqdm_bar:
            from tqdm import tqdm
            iterable = tqdm(self.df_preds.index)
        else:
            iterable = self.df_preds.index

        for i in iterable:
            if pd.notna(self.df_preds.loc[i, 'Token_Counter']):
                self.df_preds.loc[i, 'Token_Head'] = preds[counter]
                counter += 1

        rows = []
        for i, row in self.df_preds.iterrows():
            if pd.notna(row.values[0]):
                rows.append('\t'.join([str(int(col) if isinstance(col, float) else col) for col in list(row.values)]) + '\n')
            else:
                rows.append('\n')

        with open(os.path.join('preds', f'{name}_321128258.labeled'), 'w') as f:
            f.writelines(rows)

    def get_UAS(self):
        assert self.df_preds is not None, 'need to run insert_predictions first'
        assert self.df_preds.shape == self.df.shape, 'self.df_preds.shape must match self.df.shape'
        return (self.df_preds['Token_Head'].dropna().values == self.df['Token_Head'].dropna().values).mean()


