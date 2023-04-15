import numpy as np
import torch
def normalize_np(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_tensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))