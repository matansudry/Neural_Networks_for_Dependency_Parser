import pandas as pd


columns = ['Token_Counter', 'Token', 'Token_POS', 'Token_Head', 'Dependency_Label']

class DataSet:
    def __init__(self, path, columns=columns):
        self.df = pd.read_csv('data/test.labeled',
                              sep='\t',
                              names=['Token_Counter', 'Token', 3, 'Token_POS', 5 ,6 , 'Token_Head', 'Dependency_Label', 9, 10],
                              skip_blank_lines=False)#.astype({
#             'Token_Counter': int,
#             'Token_Head': int
#         })
                                                             
        if columns:
            self.df = self.df[columns]
            
    @property
    def y(self):
        return self.df['Token_Head']