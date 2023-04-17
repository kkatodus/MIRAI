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

From this data collection pipeline, we are able to generate approximately 20,000 batches of 20 frames 64x64 images for traning and testing. Note that the Video-GPT uses 16 frames. 


## Results

![Results](https://cdn.discordapp.com/attachments/1068309893171384330/1097574229471412244/image.png)

![Table](https://cdn.discordapp.com/attachments/1068309893171384330/1097565990767841401/image.png)

## Hyperparameter Tuning
As the pre-trained weights are specific to the model architecture, to be able to use the pre-trained weights, the model architecture needs to be the same. Hence we tried to experiment with the hyperparameters that doesn't change the model architecture including the batch size and the number of conditioned frames. 
### Batch Size
Due to the memory constraint, increasing the batch size requires us to change the size of the input and output to fewer frames. Our experiment found that having the batch size of 1 and have 16 frames performs the best.

![BatchSizeResults](https://cdn.discordapp.com/attachments/1068309893171384330/1097567920466427965/image.png)


### Number of Conditioned Frames
Increasing the number of conditioned frame increases the model's performance. In practice, we want to minimize the number of frames the model conditions on while performing with an acceptable performance. 

![NumberFrameResults](https://cdn.discordapp.com/attachments/1068309893171384330/1097565316822863943/image.png)
