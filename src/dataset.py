import pandas as pd
from tqdm import tqdm
import torch


SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'

class DataSet:
    def __init__(self, path, words=None, tags=None, tqdm_bar=True, tagged=True):
        self.df = pd.read_csv(path,
                              sep='\t',
                              names=['Token_Counter', 'Token', 3, 'Token_POS', 5 ,6 , 'Token_Head', 8, 9, 10],
                              skip_blank_lines=False)
                                                             
        self.max_sentence = self.df['Token_Counter'].dropna().max()
        self.X = []
        if tagged:
            self.y = []
        if words is None:
            words = list(set(self.df['Token'].dropna().values))
            words.append(UNK)
            words = {value: i+1 for i, value in enumerate(words)}
        self.words = words
        self.inv_words = {value: key for key, value in self.words.items()}
            
        if tags is None:
            tags = list(set(self.df['Token_POS'].dropna().values))
            tags.append(UNK)
            tags = {value: i+1 for i, value in enumerate(tags)}
        self.tags = tags
        self.inv_tags = {value: key for key, value in self.tags.items()}
            
        if tqdm_bar:
            iterable = tqdm(self.df.iterrows(), total=len(self.df))
        else:
            iterable = self.df.iterrows()
        
        for i, line in iterable:
            if line.values[0] == 1.0:
                X = []
                if tagged:
                    y = []
            if pd.notna(line.values[0]):
                word = self.words[line['Token']] if line['Token'] in self.words else self.words[UNK]
                tag = self.tags[line['Token_POS']] if line['Token'] in self.tags else self.tags[UNK]
                X.append([word, tag])
                if tagged:
                    y.append(line['Token_Head'])
            elif pd.isna(line.values[0]):
                self.X.append(torch.FloatTensor(X))
                if tagged:
                    try:
                        self.y.append(torch.FloatTensor(y))
                    except Exception as e:
                        print(y)
                        raise e
                    
        del self.df
        
    def __iter__(self):
        for x, y in zip(self.X, self.y):
            yield x, y