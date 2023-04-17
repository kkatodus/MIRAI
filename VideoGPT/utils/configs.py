import os

DATA_DIR = "data"
SERIES_DIR = ["Avatar_training", "Avatar_testing"]
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
LOG_DIR = "logs"

#Training parameters
EPOCHS = 1000

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)