import pandas as pd


SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

class DataSet:
    def __init__(self, path, words=None, tags=None):
        self.df = pd.read_csv('data/test.labeled',
                              sep='\t',
                              names=['Token_Counter', 'Token', 3, 'Token_POS', 5 ,6 , 'Token_Head', 8, 9, 10],
                              skip_blank_lines=False)
                                                             
#         self.max_sentence = self.df['Token_Counter'].dropna().max()
        self.X = []
        self.y = []
        if words:
            self.words = words
        else:
            self.words = {value: i+1 for i, value in enumerate(list(set(self.df['Token'].dropna().values)))}
            
        if tags:
            self.tags = tags
        else:
            self.tags = {value: i+1 for i, value in enumerate(list(set(self.df['Token_POS'].dropna().values)))}
        for i, line in tqdm(self.df.iterrows(), total=len(self.df)):
            if line.values[0] == 1.0:
                X = []
                y = []
            if pd.notna(line.values[0]):
                word = self.words[line['Token']] if line['Token'] in self.words else len(self.words)+1
                tag = self.tags[line['Token_POS']] if line['Token'] in self.tags else len(self.tags)+1
                X.append([word, tag])
                y.append(line['Token_Head'])
            elif pd.isna(line.values[0]):
                self.X.append(torch.FloatTensor(X))
                self.y.append(torch.FloatTensor(y))
        del self.df
        
    def __iter__(self):
        for x, y in zip(self.X, self.y):
            yield x, y