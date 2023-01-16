import numpy as np
import os
from torch.utils.data import Dataset
import torch


class Train_Data(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.train_X, self.train_Y = self.load_mnist_2d(data_dir)
        self.train_len = len(self.train_X)


    def load_mnist_2d(self, data_dir):
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

        trX = (trX - 128.0) / 255.0

        return torch.Tensor(trX), torch.Tensor(trY).int()


    def __getitem__(self, index):
        return self.train_X[index], self.train_Y[index]

 
    def __len__(self):
        return self.train_len


class Test_Data(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.test_X, self.test_Y = self.load_mnist_2d(data_dir)
        self.test_len = len(self.test_X)


    def load_mnist_2d(self, data_dir):
        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

        teX = (teX - 128.0) / 255.0

        return  torch.Tensor(teX), torch.Tensor (teY).int()


    def __getitem__(self, index):
        return self.test_X[index], self.test_Y[index]

 
    def __len__(self):
        return self.test_len


def onehot_encoding(label, class_num, device):
    ones = torch.sparse.torch.eye(class_num, device=device)
    return ones.index_select(0,label)