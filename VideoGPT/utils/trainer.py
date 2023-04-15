from tqdm import tqdm
import os
import numpy as np
import torch

from utils.visualizer import visualize_np_sequence_opencv, print_data_format
from utils.dataset import Dataset
from utils.configs import DATA_DIR, CHECKPOINT_DIR
from utils.calc import normalize_tensor

def train_gpt_on_files(gpt, job, device=torch.device('cuda')):
    job_name = job.job_name
    file_paths = job.file_paths
    epochs = job.epochs
    print("starting job", job_name)
    losses = []
    for _ in range(epochs):
        for file_idx, file_path in enumerate(tqdm(file_paths)):
            #getting one sample data
            file_data_np = np.load(file_path)
            file_data_np = torch.from_numpy(file_data_np).float().to(device)
            #only accepts 16 frames and 64x64 resolution
            file_data_np = file_data_np[:, :, :16]
            print_data_format(file_data_np)
            #first batch, all colors, all frames, cropped to 64x64
            #B x C x T x H x W
            #creating dataset and dataloader
            dataset = Dataset(file_data_np)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
            #getting optimizer for model
            optimizer, _ = gpt.configure_optimizers()
        
            for i, batch in enumerate(dataloader):
                # print("batch", i)
                #normalizing the batch
                batch = normalize_tensor(batch)
                input_dict = {'video': batch}

                loss, logits =gpt.training_step(input_dict, 0)
                print(loss)
                print(logits)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
    print("Done training of job:", job_name)
    print("Saving checkpoint....")
    saving_path = os.path.join(CHECKPOINT_DIR, f"job_{job_name}_epochnum_{epochs}_model.pt")
    torch.save(gpt.state_dict(), saving_path)
    return gpt, losses