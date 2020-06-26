import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm



ROOT = '<root>'
UNK = '<unk>'
PAD = '<pad>'
y_PAD = '<y_pad>'
SPECIAL = [y_PAD, PAD, UNK, ROOT]

glove_dim = 300

class DataSet:
    """
    Attributes
    -------
    df : pd.DataFrame, optional if del_df=False
        the original csv
    max_sentence_len : int
        max length of a sentence, used as the 2nd dimention of self.X, self.y
    X : torch.FloatTensor
        Token and Token_POS tensor, torch.Size([num_of_sentences, max_sentence_len, 2])
    y : torch.FloatTensor
        Token_Head tensor, torch.Size([num_of_sentences, max_sentence_len, 1])
    tokens : dict
         key=token, value=index
    inv_tokens : dict
        key=index, value=token
    """
    def __init__(self, path, train_dataset=None, tagged=True, tqdm_bar=False, use_glove=False):
        """
        Parameters
        -------
        path : str
            path to csv
        train_dataset : DataSet, optional
            shouldbe passed for test/validation datasets (default is None)
        tagged : bool, optional
            if dataset has tags (default is True)
        tqdm_bar : bool, optional
            if to display progress bar (default is False)
        keep_df : bool, optional
            if to keep original csv dataframe (default is False)
        pad : bool, optional
            if to use a large padded tensor or a list of sentence tensors (default is True)
        """
        # load csv to pd.DataFrame
        self.df = pd.read_csv(path, names=['Token_Counter', 'Token', 3, 'Token_POS', 5 ,6 , 'Token_Head', 8, 9, 10],
                              sep='\t', skip_blank_lines=False)
        self.df_preds = None
        
        if train_dataset is not None:
            if use_glove:
                assert train_dataset.embeds is True, 'train_dataset must have glove embedings'
                self.embeds = True
                self.words_vectors = train_dataset.words_vectors
            self.max_sentence_len = train_dataset.max_sentence_len

            self.words_dict = train_dataset.words_dict
            self.tags_dict = train_dataset.tags_dict
            self.special_dict = train_dataset.special_dict

            self.words_num = train_dataset.words_num
            self.tags_num = train_dataset.tags_num
        else:
            if use_glove:
                self.embeds = True
                from collections import Counter
                from torchtext.vocab import Vocab

                glove = Vocab(Counter(dict(self.df['Token'].value_counts())), vectors="glove.6B.300d", specials=SPECIAL)
                self.words_dict = glove.stoi
                self.words_vectors = glove.vectors
            else:
                self.embeds = False
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
        
        if use_glove:
            word_vectors_tensor = torch.zeros([self.words_tensor.shape[0], self.words_tensor.shape[1], glove_dim])
            for i in tqdm(range(self.words_tensor.shape[0])):
                for j in range(self.words_tensor.shape[1]):
                    word_vectors_tensor[i, j, :] = self.words_vectors[self.words_tensor[i, j]]
            self.words_tensor = word_vectors_tensor
        
        self.tags_tensor = nn.utils.rnn.pad_sequence(self.tags_tensor, batch_first=False, padding_value=self.special_dict[PAD]).transpose(1, 0)
        self.y = nn.utils.rnn.pad_sequence(self.y, batch_first=False, padding_value=self.special_dict[y_PAD]).transpose(1, 0)

    def insert_predictions(self, preds, name):
        self.df_preds = self.df.copy()
        counter = 0
        for i in tqdm(self.df_preds.index):
            if pd.notna(self.df_preds.loc[i, 'Token_Counter']):
                self.df_preds.loc[i, 'Token_Head'] = preds[counter]
                counter += 1
                
        self.df_preds.to_csv(os.path.join('preds', f'{name}_321128258.labeled'), sep='\t', index=False)

    def get_UAS(self):
        assert self.df_preds is not None, 'need to run insert_predictions first'
        assert self.df_preds.shape == self.df.shape, 'self.df_preds.shape must match self.df.shape'
        return (self.df_preds['Token_Head'].dropna().values == self.df['Token_Head'].dropna().values).mean()
                
    @property
    def dataset(self):
        return list(zip(self.words_tensor, self.tags_tensor, self.lens, self.y))

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for x, y in zip(self.X, self.y):
            yield x, y


