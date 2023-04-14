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
We have created these scripts for the data collection pipeline:
* `generate_timestamps.py` generates the timestamps of the beginning of a cut that lasts longer than 20 frames (we discard any cut that is smaller than 20 frames so that the cuts are meaningful)
* `split_video.py` split the video into cuts with 20 frames images and save them
* `generate_npy_from_jpeg.py` converts the jpeg files into a single npy file and also resize the image to the desired size. For the project we convert the image into 64x64 pixels.
*  `visualize.py` converts the numpy array back into image and videos for visualizing the results

General Data Collection Pipeline:
![Data Collection Pipeline](https://cdn.discordapp.com/attachments/1068310042908041297/1096509083785383946/data_processing.png)

From this data collection pipeline, we are able to generate approximately 20,000 batches of 20 frames 64x64 images for traning and testing.


## Results

The sample input sequence is PUT SAMPLE GIF

The sample output sequence is PUT OUTPUT GIF

These two sequences have the MSE of ....

Before fine-tuning the model, the output is PUT OUTPUT GIF Before Fine tuning