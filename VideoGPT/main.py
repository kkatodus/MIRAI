from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import torchmetrics

import os
import torch
from torchvision.io import read_video, read_video_timestamps
import numpy as np

from videogpt import download, load_vqvae, load_videogpt
from videogpt import download, load_videogpt
from videogpt.data import preprocess

import itertools


DATA_DIR = "VideoGPT/data"

def main():

    #loading the video GPT model
    device = torch.device('cuda')
    gpt = load_videogpt('bair_gpt', device=device).to(device)
    #getting paths to data    
    files = os.listdir(DATA_DIR)
    files = [os.path.join(DATA_DIR, f) for f in files]
    #getting one sample data
    sample_data_np = np.load(files[0])
    sample_data_np = torch.from_numpy(sample_data_np).float().to(device)
    #first batch, all colors, all frames, cropped to 64x64
    #B x C x T x H x W
    sample_data_np = sample_data_np[:1, :, :, 64:, 64:]
    print("Sample Input Shape is:", sample_data_np.shape)
    sample_input_dict = {'video': sample_data_np}
    gpt.args.n_cond_frames = 10
    print("We are conditioning on the first n frames", gpt.args.n_cond_frames)
    
    gpt.training_step(sample_input_dict, 0)
    

    pass

if __name__ == '__main__':
    main()