import torch
import numpy as np
import pandas as pd


# SOS = '<SOS>'
# EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
SPECIAL = [PAD, UNK, ]

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
    words : dict
         key=word, value=index
    inv_words : dict
        key=index, value=word
    tags : dict
        key=tag, value=index
    inv_tags : dict
        key=index, value=tag
    """
    def __init__(self, path, train_dataset=None, tagged=True, tqdm_bar=False, keep_df=False, pad=True):
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

        if train_dataset is None:
            # self.max_sentence_len is used as self.X, self.y 2nd dimention
            self.max_sentence_len = self.df['Token_Counter'].dropna().max() + 1
            self.token_heads = [i - len(SPECIAL) for i in list(range(int(self.max_sentence_len)))]

            # init words + inv_words dicts
            # SPECIAL tokens will have negative indecies
            words = SPECIAL + list(set(self.df['Token'].dropna().values))
            self.words = {value: float(i) - len(SPECIAL) for i, value in enumerate(words)}
#             self.words = {value: float(i) for i, value in enumerate(words)}
            self.inv_words = {value: key for key, value in self.words.items()}

            # init tags + inv_tsgs dicts
            # SPECIAL tokens will have negative indecies
            tags = SPECIAL + list(set(self.df['Token_POS'].dropna().values))
            self.tags = {value: float(i) - len(SPECIAL) for i, value in enumerate(tags)}
#             self.tags = {value: float(i) for i, value in enumerate(tags)}
            self.inv_tags = {value: key for key, value in self.tags.items()}
            
            self.special = {key: self.words[key] for key in SPECIAL}
        else:
            self.max_sentence_len = train_dataset.max_sentence_len
            self.token_heads = train_dataset.token_heads

            self.words = train_dataset.words
            self.inv_words = train_dataset.inv_words
            
            self.tags = train_dataset.tags
            self.inv_tags = train_dataset.inv_tags

            self.special = train_dataset.special
            
        # fill X, y tensors with data
        self.X = []
        if tagged:
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
                if pad:
                    sentence_X = np.zeros([int(self.max_sentence_len), 2]) + self.words[PAD]
                else:
                    sentence_X = []
                if tagged:
                    if pad:
                        sentence_y = np.zeros([int(self.max_sentence_len), 1]) + self.words[PAD]
                    else:
                        sentence_y = []

            if pd.notna(i):
                word = self.words[line['Token']] if line['Token'] in self.words else self.words[UNK]
                tag = self.tags[line['Token_POS']] if line['Token_POS'] in self.tags else self.tags[UNK]
                if pad:
                    sentence_X[int(i) - 1] = [word, tag]
                    if tagged:
                        sentence_y[int(i) - 1] = line['Token_Head']
                else:
                    sentence_X.append([word, tag])
                    if tagged:
                        sentence_y.append(line['Token_Head'])
                    
            else:
                self.X.append(torch.LongTensor(sentence_X))
                if tagged:
                    self.y.append(torch.LongTensor(sentence_y))
        
        # convert X, y to tensors
        if pad:
            self.X = torch.LongTensor(self.X)
            if tagged:
                self.y = torch.LongTensor(self.y)
            else:
                self.y = torch.LongTensor(np.zeros(list(self.X.shape[0:2]) + [1]))
            self.y = torch.squeeze(self.y, dim=2)

        if not keep_df:
            del self.df

    def __len__(self):
        return len(self.X)
            
    def __iter__(self):
        for x, y in zip(self.X, self.y):
            yield x, y


