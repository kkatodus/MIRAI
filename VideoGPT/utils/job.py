import os
class Job:
    def __init__(self, data_dir, data_path, job_name, epochs=1):
        self.data_dir = data_dir
        self.data_path = data_path
        if data_dir: 
            self.file_paths = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir)]
        else:
            self.file_paths = [data_path]
        self.job_name = job_name
        self.epochs = epochs


    def __str__(self):
        return f"Job: {self.job_name} | Data dir: {self.data_dir} | Data path: {self.data_path} |"
    