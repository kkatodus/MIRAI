"""
Utility functions for converting data from img and text to numpy arrays and text files
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import ffmpeg
import ast
import subprocess
import os
from pathlib import Path
from scipy.ndimage import zoom


def jpg_to_np(path):
    """
    This function converts the jpg images to numpy arrays
    :param path: the path to the image
    :return: the numpy array of the image
    """
    img = Image.open(path)
    img = np.array(img)
    return img


def get_one_cut(path):
    """
    This function gets one cut of the video and return a list of numpy array and the text file
    :param path: the path to the folder containing a jpg image and a text file
    :return: the numpy array of the video and the text file
    """
    folder = os.listdir(path)
    list_of_frames = []
    text = ""
    # iterate through all files in the folder
    frame_count = 0
    file_name = []
    for file in folder:
        if file.endswith(".jpg"):
            frame_count += 1
            file_name.append(file)
            #img = jpg_to_np(path + "/" + file)
            # list_of_frames.append(img)
        elif file.endswith(".txt"):
            # found a text file
            text = open(path + "/" + file, "r")
            text = text.read()
    file_name.sort()
    for i in range(frame_count):
        img = jpg_to_np(path + "/" + file_name[i])
        list_of_frames.append(img)

    return list_of_frames, text


def save_frames_to_file(path, list_of_frames):
    """ This function saves the list_of_frames to a file as 4D numpy array
    """
    array = np.array(list_of_frames)
    np.save(path, array)
    return array


def check_frame_difference(first_frame, second_frame, threshold=0.65):
    """ this function checks whether the difference between the two frames is greater than the threshold
    """
    """
    # compute the absolute difference between the two frames
    diff = cv2.absdiff(first_frame, second_frame)
    # print("diff: ", diff)

    # convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # create a binary image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    """

    # this part uses SSIM to compare the two frames
    gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    ssim = compare_ssim(gray1, gray2)
    # print("ssim", ssim)
    return ssim < threshold


def get_cut_timestamps(video_path, threshold=0.65):

    # load video
    cap = cv2.VideoCapture(video_path)

    # check if it has loaded
    if not cap.isOpened():
        print("Error opening video stream or file")

    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get first frame
    ret, first_frame = cap.read()

    # initialize list of timestamps
    timestamps = []
    n = 0
    # iterate through all frames
    while ret:
        n += 1
        ret, frame = cap.read()
        cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if ret:
            # print("frame number: ", n)
            # print("cur_time: ", cur_time)
            if check_frame_difference(first_frame, frame, threshold) > 0:
                # if the difference is greater than the threshold, add the timestamp to the list
                cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                timestamps.append(cur_time)

            first_frame = frame
        else:
            print("Break from the while loop at cur_time: ", cur_time)
            break

    cap.release()
    return timestamps


def save_list_to_file(list, path):
    """ This function saves the list as the path
    """
    f = open(path, "w")
    f.write(str(list))
    f.close()


def remove_consecutive_timestamps(timestamps):
    """ This function removes consecutive timestamps
    """
    new_list = []
    for i in range(len(timestamps)):
        if i == 0:
            new_list.append(timestamps[i])
        else:
            if timestamps[i] - new_list[-1] >= 2:
                new_list.append(timestamps[i])
            else:
                # remove the last timestamp
                new_list.remove(new_list[-1])
                new_list.append(timestamps[i])
    return new_list


def split_to_images(video_path, output_path, timestamps, quality=2, fps=10):
    # Define the start and end time in seconds
    input_file = video_path
    output_dir = output_path

    input_stream = ffmpeg.input(input_file)

    for i in range(len(timestamps)-1):
        start_time = timestamps[i]
        end_time = timestamps[i] + 2
        output_dir = output_path
        if not os.path.exists(output_dir + str(i)):
            os.makedirs(output_dir + str(i))
        output_dir = output_dir + str(i)+"/"
        output_stream = (input_stream
                         .trim(start=start_time, end=end_time)
                         .setpts('PTS-STARTPTS')
                         .filter('fps', fps=fps)
                         .output(f"{output_dir}/frame_%04d.jpg", codec='mjpeg',
                                 qscale=quality)
                         )
        ffmpeg.run(output_stream)


def reshape(data, num_frame, height, width):
    """ This function reshapes the data to the correct shape
    """
    height_low = int((data[0].shape[0] - height)/2)
    height_high = height_low + height
    width_low = int((data[0].shape[1] - width)/2)
    width_high = width_low + width
    return np.array(data)[:num_frame, height_low:height_high, width_low:width_high, :]


def scale(data, original_height, original_width, new_height, new_width):
    y_scale = new_height / original_height
    x_scale = new_width / original_width
    channel_scale = 1
    resized_data = np.empty(
        (data.shape[0], new_height, new_width, data.shape[3]), dtype=np.uint8)
    for i in range(data.shape[0]):
        resized_float = zoom(data[i], (y_scale, x_scale, channel_scale))
        resized_uint8 = ((resized_float - np.min(resized_float)) / (
            np.max(resized_float) - np.min(resized_float) + 1e-6) * 255).astype(np.uint8)
        # print("inside scale: ", resized_uint8.dtype)
        resized_data[i] = resized_uint8
    return resized_data


def reshape_specify_position(data, num_frame, height_low, height_high, width_low, width_high):
    """ This function reshapes the data to the correct shape
    """
    # height_low = int((data[0].shape[0] - height)/2)
    # height_high = height_low + height
    # width_low = int((data[0].shape[1] - width)/2)
    # width_high = width_low + width
    return np.array(data)[:num_frame, height_low:height_high, width_low:width_high, :]


if __name__ == "__main__":
    """
    for i in range(30):
        path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/autogenerated/" + \
            str(i)
        list_of_frames, text = get_one_cut(path)
        numpy_path = path + "/" + str(i) + ".npy"
        save_frames_to_file(path, list_of_frames)


    with open("/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_timestamps.txt", "r") as f:
        text = f.read()
    timestamps = ast.literal_eval(text)
    output_path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/auto_images/"
    split_to_images(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", output_path, timestamps)


    threshold = 0.7
    print("Threshold: ", threshold)
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/videos/out_of_sight_1"
    list_of_frames, text = get_one_cut(path)
    print(len(list_of_frames))
    timestamps = get_cut_timestamps(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", threshold)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    timestamps = remove_consecutive_timestamps(timestamps)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    save_list_to_file(
        timestamps, "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_timestamps.txt")
    """
    """
    array = save_to_file(
        "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_1.npy", list_of_frames)
    print(array.shape)
    """
