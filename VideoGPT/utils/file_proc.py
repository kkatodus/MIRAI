import os
def get_newest_file_in_dir(dir_path):
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def get_oldest_file_in_dir(dir_path):
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, basename) for basename in files]
    return min(paths, key=os.path.getctime)

def write_log(log_file_path, message):
    file = open(log_file_path, "a")
    file.write(message)
    file.close()
