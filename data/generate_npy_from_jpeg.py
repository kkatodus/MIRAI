import util
import numpy as np
import os

if __name__ == "__main__":
    save_path = "/Volumes/OneTouch/Totoro/totoro_scaled"

    path = "/Volumes/OneTouch/Totoro/cuts90/"
    print("Generate from: ", path)
    res = []
    # list_of_frames, text = util.get_one_cut(path)
    # resized_list = util.reshape(list_of_frames, 20, 128, 128)
    # array = np.array(resized_list)

    num_cuts = len(os.listdir(path)) - 1
    for i in range(num_cuts):
        cut_path = path + \
            str(i)
        list_of_frames, text = util.get_one_cut(cut_path)
        # print("len(list_of_frames[0][0]): ", len(list_of_frames[0][0]))
        height = len(list_of_frames[0])
        resized_list = util.reshape(list_of_frames, 20, height, height)
        # print("resized_list.shape: ", resized_list.shape)
        scaled_list = util.scale(resized_list, height, height, 64, 64)
        # print("data_type: ", scaled_list.dtype)
        scaled_list = scaled_list.transpose(3, 0, 1, 2)
        # resized_list = np.expand_dims(resized_list, axis=0)
        res.append(scaled_list)
    array = np.array(res)
    np_path = save_path+".npy"
    print("save at: ", np_path)
    print("array.shape", array.shape)
    np.save(np_path, array)
    """
    for j in range(1, 21):
        path = "/Volumes/OneTouch/Avatar/avatar2_" + \
            str(j) + "/"
        print("Generate from: ", path)
        res = []
        # list_of_frames, text = util.get_one_cut(path)
        # resized_list = util.reshape(list_of_frames, 20, 128, 128)
        # array = np.array(resized_list)

        num_cuts = len(os.listdir(path)) - 1
        for i in range(num_cuts):
            cut_path = path + \
                str(i)
            list_of_frames, text = util.get_one_cut(cut_path)
            resized_list = util.reshape(list_of_frames, 20, 64, 64)
            resized_list = resized_list.transpose(3, 0, 1, 2)
            # resized_list = np.expand_dims(resized_list, axis=0)
            res.append(resized_list)
        array = np.array(res)
        np_path = save_path+"2_"+str(j)+".npy"
        print("save at: ", np_path)
        print("array.shape", array.shape)
        np.save(np_path, array)
        """
