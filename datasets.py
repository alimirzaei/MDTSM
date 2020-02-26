from torch.utils.data import Dataset
import numpy as np
import torch

class SyntethicData(Dataset):
    def __init__(self):
        dataset = np.load('lstm_data.npz')
        self.X = torch.from_numpy(dataset['X_train'])
        self.Y = torch.from_numpy(dataset['Y_train'])
        self.n_samples = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.n_samples
