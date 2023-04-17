#%%
from matplotlib import pyplot as plt

from tqdm import tqdm
import os
import torch
import numpy as np

from videogpt import load_videogpt
from videogpt import load_videogpt
from datetime import datetime
import time

from utils.visualizer import visualize_np_sequence_opencv, print_data_format
from utils.configs import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR, EPOCHS
from utils.job import Job
from utils.dataset import Dataset, is_bad_input
from utils.file_proc import get_newest_file_in_dir, get_oldest_file_in_dir, write_log
from utils.calc import normalize_np, normalize_tensor


#%%
def fix_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

#loading the model and file destinations
import random
#loading the video GPT model
device = torch.device('cuda')
gpt = load_videogpt('bair_gpt', device=device).to(device)
gpt.args.n_cond_frames = 1
gpt.args.max_steps = 10000
optimizer, _ = gpt.configure_optimizers()


#%%
# this code was used to split the data into training and testing sets and save them as files
# for entry in os.listdir(DATA_DIR):
#     if entry=="yourname64_scaled.npy":
#         path = os.path.join(DATA_DIR, entry)
#         data_np = np.load(path)
#         np.random.shuffle(data_np)
#         batches = data_np.shape[0]
#         train_data = data_np[:int(batches*0.8)]
#         test_data = data_np[int(batches*0.8):]
#         np.save(os.path.join(DATA_DIR, "train_" + entry), train_data)
#         np.save(os.path.join(DATA_DIR, "test_" + entry), test_data)

#         print(data_np.shape)
#%%
def train_gpt_on_files(job, device=torch.device('cuda'), epoch = 0):
    gpt.train()
    job_name = job.job_name
    file_paths = job.file_paths
    print("starting job", job_name)
    losses = []

    for file_idx, file_path in enumerate(tqdm(file_paths)):
        write_log(os.path.join(LOG_DIR, "log.txt"), f"Starting file {file_idx} of {len(file_paths)} in job {job_name} at {str(datetime.now())}\n")
        #getting one sample data
        file_data_np = np.load(file_path)
        file_data_np = torch.from_numpy(file_data_np).float().to(device)
        #only accepts 16 frames and 64x64 resolution
        file_data_np = file_data_np[:, :, :16]
        #first batch, all colors, all frames, cropped to 64x64
        #B x C x T x H x W
        #creating dataset and dataloader
        dataset = Dataset(file_data_np)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for i, batch in enumerate(dataloader):
            
            # print("batch", i)
            #normalizing the batch
            # batch = normalize_tensor(batch)
            batch /= 255
            #making it between -0.5 and 0.5
            # batch -= 0.5
            if is_bad_input(batch):
                print("bad input found in batch:", i, "of", file_path)
                continue
            input_dict = {'video': batch}

            loss, logits =gpt.training_step(input_dict, 0)
            if torch.isnan(logits).any():
                raise Exception(f"\nNan found in logits:{i}th batch of {file_path}")

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
    
    print("Done training of job:", job_name)
    print("Saving checkpoint....")
    now = datetime.now()    
    current_time = now.strftime("day_%d_%m_%y_time_%H_%M_%S")
    saving_path = os.path.join(CHECKPOINT_DIR, f"job_{job_name.split('.')[0]}_epoch_{epoch}_time_{current_time}.pt")
    torch.save(gpt.state_dict(), saving_path)
    
    return losses

#%%
#prepare jobs for the training
jobs = []
directory = os.listdir(DATA_DIR)
directory = ["train_yourname64_scaled.npy"]
for entry in directory:
    path_to_entry = os.path.join(DATA_DIR, entry)
    if not entry.startswith("train_"):
        continue
    #one directory contains multiple npy files for training
    if os.path.isdir(path_to_entry):
        job = Job(path_to_entry, None, entry, epochs=EPOCHS)
    elif entry.endswith(".npy"):
        job = Job(None, path_to_entry, entry, epochs=EPOCHS)
    jobs.append(job)

print("---------------------------------------")
print("REGISTERED JOBS:")
print("---------------------------------------")
for job in jobs:
    print(job)
print("---------------------------------------")
print("---------------------------------------")

#log the jobs
#open log file
#%%
#training loop
losses = []
if len(os.listdir(CHECKPOINT_DIR)) == 0:
    print("No checkpoints found, starting from scratch")
else:
    print("Found checkpoints, loading them")
    checkpoint_path = os.path.join("checkpoints", "job_train_yourname64_scaled_epoch_1_time_day_15_04_23_time_20_59_16.pt")
    print("Loading checkpoint", checkpoint_path)
    gpt.load_state_dict(torch.load(checkpoint_path))
for epoch in range(EPOCHS):
    for job in jobs:
        write_log(os.path.join(LOG_DIR, "log.txt"), f"\n\nstarting epoch {epoch} of job {job.job_name} at {str(datetime.now())}\n-------------------------------------------------------------------------\n")
        losses = train_gpt_on_files(job,device, epoch)
        write_log(os.path.join(LOG_DIR, "log.txt"), f"finished epoch {epoch} of job {job.job_name} at {str(datetime.now())}\n-------------------------------------------------------------------------\n\n")
        now = datetime.now()    
        current_time = now.strftime("day_%d_%m_%y_time_%H_%M_%S")
        plt.plot(losses)
        plt.savefig(os.path.join(RESULTS_DIR, f"{job.job_name}_losses_{current_time}_epoch_{epoch}.png"))   
        plt.clf()
        plt.cla() 

#%%
#visualizing what we are feeding into the model
#get all the testing files
device = torch.device('cuda')
testing_files = [file for file in os.listdir(DATA_DIR) if file.startswith("test_") and not os.path.isdir(os.path.join(DATA_DIR, file))]

TEST_FILE_IDX = 1
test_file = testing_files[TEST_FILE_IDX]
test_file = "test_yourname64_scaled.npy"
print("testing file:", test_file)
#getting one sample data
file_data_np = np.load(os.path.join(DATA_DIR, test_file))
# file_data_np = np.load(os.path.join(DATA_DIR, "test_Avatar","2_16.npy"))
file_data_np = torch.from_numpy(file_data_np).float().to(device)
file_data_np = file_data_np[:, :, :16,]
#only accepts 16 frames and 64x64 resolution
print_data_format(file_data_np)
#B x C x T x H x W
#creating dataset
dataset = Dataset(file_data_np)
#visualize sample data
#good looking outputs
#your name and clip 5
#your name and clip 20
#your name and clip 199


CLIP_IDX = 5
sample_data = dataset[CLIP_IDX]
sample_data = sample_data.permute(1, 2, 3, 0)
sample_data = sample_data.cpu().numpy()
#values need to be between 0 and 255
sample_data = (sample_data - np.min(sample_data))*255/(np.max(sample_data) - np.min(sample_data))
sample_data = np.rint(sample_data)
sample_data = sample_data.astype(np.uint8)
np.save("sample_data.npy", sample_data)
visualize_np_sequence_opencv(sample_data, "sample_data.mp4", fps=15)


#%%
#getting output from the gpt model
gpt = load_videogpt('bair_gpt', device=device).to(device)
cond = {'video': normalize_tensor(dataset[CLIP_IDX:CLIP_IDX+1])}
#output without training
samples = gpt.sample(1, cond)
samples = samples[0].permute(1, 2, 3, 0)
samples = samples.cpu().numpy()
samples = (samples - np.min(samples))*255/(np.max(samples) - np.min(samples))
samples = np.rint(samples)
samples = samples.astype(np.uint8)
np.save("sample_out_wo_training.npy", samples)
visualize_np_sequence_opencv(samples, "model_output_wo_training.mp4", fps=15)
#output with training
#%%
checkpoint_path = os.path.join("checkpoints", "job_train_yourname64_scaled_epoch_1_time_day_15_04_23_time_20_59_16.pt")
checkpoint_path = os.path.join("checkpoints", "job_train_yourname64_scaled_epoch_137_time_day_16_04_23_time_15_23_31.pt")
print("Loading checkpoint", checkpoint_path)
checkpoint_path = get_newest_file_in_dir(CHECKPOINT_DIR)

gpt.load_state_dict(torch.load(checkpoint_path))
samples = gpt.sample(1, cond)
samples = samples[0].permute(1, 2, 3, 0)
samples = samples.cpu().numpy()

samples*=10000
#values need to be between 0 and 255
samples = (samples - np.min(samples))*255/(np.max(samples) - np.min(samples))
samples = np.rint(samples)
samples = samples.astype(np.uint8)
np.save("sample_out_w_training.npy", samples)
visualize_np_sequence_opencv(samples, "model_output_w_training.mp4", fps=15)


# %%
#getting output from the gpt model
# device = torch.device('cuda')
# gpt = load_videogpt('bair_gpt', device=device).to(device)
# saving_path = get_newest_file_in_dir(CHECKPOINT_DIR)
# torch.save(gpt.state_dict(), saving_path)
# gpt.load_state_dict(torch.load(saving_path))
# gpt.args.n_cond_frames = 10
# gpt.args.max_steps = 1000
# gpt.load_state_dict(torch.load(get_newest_file_in_dir(CHECKPOINT_DIR)))
# gpt.eval()
# input_dict = {'video': normalize_tensor(dataset[CLIP_IDX:CLIP_IDX+1])}
# gpt.training_step(input_dict, 0)