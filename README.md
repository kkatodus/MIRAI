# MIRAI: Future Frame Prediction of Anime from Previous Frames and Audio Input

This project is a part of ECE324 at the University of Toronto

## Motivation
This project involves using a machine learning model to predict the next anime frame from the previous frames and the audio of the anime. The idea is inspired by the issue in the anime industry that all of the frames need to be drawn by hand, which can make the process of creating anime expensive. In addition, some producers have used computer generated imagery to speed up the process but the viewers are not satisfied with it. 

## Architecture
![Architecture](https://cdn.discordapp.com/attachments/1036873248647942185/1096503000622698657/videoGPT.png)
This project used the same architecture as mentioned in the VideoGPT Paper as showned in the figure above. 
VideoGPT is a combination of two different models, Vector Quantized Variational Autoencoder (VQVAE) and Transformer model (GPT/Image-GPT). The VQ-VAE is used to compress video frames into discrete latent codes, which are then used as input to the Transformer model. The Transformer is then used to generate future video frames based on the compressed latent codes.

## Data Collection
The functions for converting data is under the folder `data`
The main work for data collection is done inside this folder.
We have created the three scripts for the data collection pipeline:
* `generate_timestamps.py` generates the timestamps of the beginning of a cut that lasts longer than 20 frames (we discard any cut that is smaller than 20 frames so that the cuts are meaningful)
* `split_video.py` split the video into cuts with 20 frames images and save them
* `generate_npy_from_jpeg.py` converts the jpeg files into a single npy file and also resize the image to the desired size. For the project we convert the image into 64x64 pixels.

General Data Collection Pipeline:
![Data Collection Pipeline](https://cdn.discordapp.com/attachments/1068310042908041297/1096505044553183263/data_processing.png)

## Data Visualization
Data visualization involves two functions that convert the numpy arrays back into video cuts. The functions are inside the file `/data_proc/data_proc.py`


## VATT
The code for running VATT encoder is under the folder `VATT`
VATT model is configured to the configuration that runs with out input. The configuration is set in `main.py` inside the root folder. The dataloader that VATT required does not work with our dataset, hence needs to overwrite through the configurations.

## LSTM Decoder
A 2D convLSTM is defined in `main.py`
The output from the VATT encoder is passed into the first hidden state of the decoder. Without training, the output was shown in the presentation and the report. The sample video is : [need a link]
