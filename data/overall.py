import util


if __name__ == "__main__":
    # Get timestamps
    # note: higher threshold means that the video will be split into more parts
    threshold = 0.88
    print("Threshold: ", threshold)
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/videos/out_of_sight_1"
    list_of_frames, text = util.get_one_cut(path)
    print(len(list_of_frames))
    timestamps = util.get_cut_timestamps(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", threshold)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    timestamps = util.remove_consecutive_timestamps(timestamps)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    util.save_list_to_file(
        timestamps, "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_timestamps.txt")

    # Split video into images
    with open("/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_timestamps.txt", "r") as f:
        text = f.read()
    timestamps = ast.literal_eval(text)
    output_path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/auto_images/"
    util.split_to_images(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", output_path, timestamps)
