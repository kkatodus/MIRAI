import cv2
import os 
from PIL import Image

def visualize_np_sequence_opencv(np_sequence, video_name="video.mp4", fps=30, dir="output"):
    """
    A function to visualize a numpy sequence of images by outputting a video corresponding to the sequence

    :param np_sequence: a numpy array of shape (number_of_frames, height, width, channels)
    :param video_name: the name of the video to be saved
    :param fps: the number of frames per second
    :return: None
    """
    np_sequence = np_sequence.transpose(1, 2, 3, 0)
    first_image = np_sequence[0]
    print("np_sequence.shape: ", np_sequence.shape)
    number_of_frames = np_sequence.shape[0]
    print("Number of frames: ", number_of_frames)
    print("Shape of first image: ", first_image.shape)
    height, width, _ = first_image.shape
    if not os.path.exists(dir):
        os.makedirs(dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{dir}/{video_name}", fourcc, fps, (width, height))
    for i in range(np_sequence.shape[0]):
        frame_bgr = cv2.cvtColor(np_sequence[i], cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    video.release()
    cv2.destroyAllWindows()

def output_images_from_np_sequence(np_sequence, output_dir="output"):
    """
    A function to output a numpy sequence of images

    :param np_sequence: a numpy array of shape (number_of_frames, height, width, channels)
    :param output_dir: the directory to output the images
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(np_sequence.shape[0]):
        image = np_sequence[i]
        pil_img = Image.fromarray(image)
        path = os.path.join(output_dir, "frame_" + str(i) + ".jpg")
        pil_img.save(path)

    