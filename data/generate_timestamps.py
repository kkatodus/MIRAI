import util

if __name__ == "__main__":
    # note: higher threshold means that the video will be split into more parts
    threshold = 0.8
    print("Start getting timestamps...")
    print("Threshold: ", threshold)
    video_path = "/Volumes/OneTouch/SpiritedAway/spiritedAway.mp4"
    print("Getting Timestamps for :", video_path)
    res_text_path = "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/spiritedAway90.txt"
    timestamps = util.get_cut_timestamps(
        video_path, threshold)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    timestamps = util.remove_consecutive_timestamps(timestamps)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    util.save_list_to_file(
        timestamps, res_text_path)
