from torch.utils.data import Dataset
import os

class TripleSets(Dataset):
    def __init__(self, path='../input/mkgsets'):
        sets = ['train', 'valid']
        path1 = os.path.join(path, '{}2id.txt'.format(sets[0]))
        path2 = os.path.join(path, '{}2id.txt'.format(sets[1]))
        f1 = open(path1, 'r')
        f2 = open(path2, 'r')
        data1 = f1.readlines()[1:]
        data1 = [line.split() for line in data1]
        data1 = [[int(num) for num in line] for line in data1]
        data2 = f2.readlines()[1:]
        data2 = [line.split() for line in data2]
        data2 = [[int(num) for num in line] for line in data2]
        self.data = data1 + data2
    
    def __getitem__(self, index):
        item = self.data[index]
        return item[0], item[2], item[1]
    
    def __len__(self):
        return len(self.data)