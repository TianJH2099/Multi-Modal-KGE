from torch.utils.data import Dataset
import os
import pandas as pd

class TripleSets(Dataset):
    def __init__(self, path='OpenBG-IMG', c="train"):

        f_path = os.path.join(path, '{}2id.tsv'.format(c))
        f = pd.read_csv(f_path, sep='\t')
        self.data = f
    
    def __getitem__(self, index):
        item = self.data.loc[index]
        return item[0], item[1], item[2]
    
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    mySet = TripleSets()
    print(mySet.__getitem__(1))
    print(mySet.__len__())