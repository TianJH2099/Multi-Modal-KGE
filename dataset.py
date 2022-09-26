from torch.utils.data import Dataset
import os

class TripleSets(Dataset):
    def __init__(self, path='OpenBG-IMG', c="train"):

        f_path = os.path.join(path, '{}2id.tsv'.format(c))
        with open(f_path, 'r') as f:
            data = f.readlines()[0:]
            data = [line.split() for line in data]
            data = [[int(num) for num in line] for line in data]
        self.data = data
    
    def __getitem__(self, index):
        item = self.data[index]
        return item[0], item[2], item[1]
    
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    mySet = TripleSets()
    print(mySet.__getitem__(1))
    print(mySet.__len__())