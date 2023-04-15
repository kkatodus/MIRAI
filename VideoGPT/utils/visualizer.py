import os 
import cv2
import numpy as np

def visualize_np_sequence_opencv(np_sequence, video_name="video.mp4", fps=30, dir="output"):
    """
    A function to visualize a numpy sequence of images by outputting a video corresponding to the sequence
    NOTE: The values need to be between 0 and 255
    NOTE: make sure the input is all int values
    :param np_sequence: a numpy array of shape (number_of_frames, height, width, channels)
    :param video_name: the name of the video to be saved
    :param fps: the number of frames per second
    :return: None
    """
    np_sequence = np.rint(np_sequence).astype(np.uint8)
    if np.max(np_sequence) > 255 or np.min(np_sequence) < 0:
        raise ValueError("The values need to be between 0 and 255")
    first_image = np_sequence[0]
    number_of_frames = np_sequence.shape[0]
    # print("Number of frames: ", number_of_frames)
    # print("Shape of first image: ", first_image.shape)
    height, width, _ = first_image.shape
    if not os.path.exists(dir):
        os.makedirs(dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{dir}/{video_name}", fourcc, fps, (width, height))
    for i in range(np_sequence.shape[0]):
        video.write(np_sequence[i])
    video.release()
    cv2.destroyAllWindows()


def print_data_format(data):
    if len(list(data.shape)) != 5:
        print("Data must be in B x C x T x H x W format")
    else:
        print("batch size", data.shape[0])
        print("channels", data.shape[1])
        print("frames", data.shape[2])
        print("height", data.shape[3])
        print("width", data.shape[4])