import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def is_bad_input(array):
    flag = False
    if torch.sum(array)==0:
        print("Batch sums to zero")
        flag = True
    if torch.isnan(array).any():
        print("Nan found in batch")
        flag = True
    return flag