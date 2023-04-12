#%%
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import torchmetrics

from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from torchvision.io import read_video, read_video_timestamps
import numpy as np

from videogpt import download, load_vqvae, load_videogpt
from videogpt import download, load_videogpt
from videogpt.data import preprocess


#%%
#prepping the dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def print_data_format(data):
    if len(list(data.shape)) != 5:
        print("Data must be in B x C x T x H x W format")
    else:
        print("batch size", data.shape[0])
        print("channels", data.shape[1])
        print("frames", data.shape[2])
        print("height", data.shape[3])
        print("width", data.shape[4])



#%%
#paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)



#%%
#loading the model and file destinations
import random
print("root dir", os.listdir("."))
#loading the video GPT model
device = torch.device('cuda')
gpt = load_videogpt('bair_gpt', device=device).to(device)
gpt.args.n_cond_frames = 10
gpt.args.max_steps = 1000
training_files_proportion = 0.8
#getting paths to data    
files = os.listdir(DATA_DIR)
files = [os.path.join(DATA_DIR, f) for f in files]
training_files = random.choices(files, k=int(len(files)*training_files_proportion))
testing_files = [f for f in files if f not in training_files]
print("training files", len(training_files))
print("testing files", len(testing_files))


#%%
#training loop
losses = []
for file in tqdm(training_files):
    print("Training file:", file)
    #getting one sample data
    file_data_np = np.load(file)
    file_data_np = torch.from_numpy(file_data_np).float().to(device)
    #only accepts 16 frames and 64x64 resolution
    file_data_np = file_data_np[:, :, :16, 64:, 64:]
    print_data_format(file_data_np)
    #first batch, all colors, all frames, cropped to 64x64
    #B x C x T x H x W
    #creating dataset and dataloader
    dataset = Dataset(file_data_np)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    #getting optimizer for model
    optimizer, _ = gpt.configure_optimizers()

    for i, batch in enumerate(dataloader):
        print("batch", i)
        input_dict = {'video': batch}

        loss, logits =gpt.training_step(input_dict, 0)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
#%%
#visualizing the output of the gpt model
import itertools
from utils.visualizer import visualize_np_sequence_opencv

#getting one sample data
file_data_np = np.load(testing_files[0])
file_data_np = torch.from_numpy(file_data_np).float().to(device)
#only accepts 16 frames and 64x64 resolution
file_data_np = file_data_np[:, :, :16, 64:, 64:]
print_data_format(file_data_np)
#first batch, all colors, all frames, cropped to 64x64
#B x C x T x H x W
#creating dataset and dataloader
dataset = Dataset(file_data_np)

#visualize sample data and original data
sample_data = dataset[:1]
#the input data needs to be in B x T x H x W x C format
sample_data = sample_data[0].permute(1, 2, 3, 0)
sample_data = sample_data.cpu().numpy()
#values need to be between 0 and 255
sample_data = (sample_data - np.min(sample_data))*255/(np.max(sample_data) - np.min(sample_data))
sample_data = np.rint(sample_data)
sample_data = sample_data.astype(np.uint8)
visualize_np_sequence_opencv(sample_data, "sample_data.mp4", fps=15)

#getting output from the gpt model
gpt.eval()
cond = {'video': sample_data}
samples = gpt.sample(1, {"video":dataset[:1]})
print("sample shape", samples.shape)
samples = samples[0].permute(1, 2, 3, 0)
samples = samples.cpu().numpy()
#values need to be between 0 and 255
samples = (samples - np.min(samples))*255/(np.max(samples) - np.min(samples))
samples = np.rint(samples)
samples = samples.astype(np.uint8)
visualize_np_sequence_opencv(samples, "model_output.mp4", fps=15)


# %%

# %%
