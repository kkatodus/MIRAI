""" This file hosts the script for evaluating the performance of the model"""

import os
import numpy as np


def mean_squared_error(img1, img2):
    """ This function calculates the mean squared error between two images
    :param img1: the first image
    :param img2: the second image
    :return: the mean squared error
    """
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def sequence_mean_squared_error(seq1, seq2):
    """This function calculates the mean squared error between two sequences
    :param seq1: the first sequence
    :param seq2: the second sequence
    :return: the mean squared error of the sequence
    """
    # the frame is the first dimension
    num_img = seq1.shape[0]
    mse = 0
    if seq1.shape != seq2.shape:
        print("The sequences are not of the same size")
        return -1
    for i in range(len(seq1)):
        mse += mean_squared_error(seq1[i], seq2[i])
    return mse / num_img


if __name__ == "__main__":
    path_to_seq1 = "/Volumes/OneTouch/Doraemon/doraemon64.npy"
    path_to_seq2 = "/Volumes/OneTouch/YourName/yourname64_scaled.npy"

    seq1 = np.load(path_to_seq1)
    seq2 = np.load(path_to_seq2)

    frame1 = np.ones((20, 128, 128, 3))
    frame2 = 10*np.ones((20, 128, 128, 3))

    mse = sequence_mean_squared_error(frame1, frame2)
    print("MSE: ", mse)
