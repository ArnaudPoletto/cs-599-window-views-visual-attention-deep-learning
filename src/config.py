import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import torch
from pathlib import Path

paths_file_path = f"{GLOBAL_DIR}/paths.py"
if not os.path.exists(paths_file_path):
    raise FileNotFoundError(f"❌ File {Path(paths_file_path).resolve()} not found. Please create it and define DATA_PATH and CODE_PATH.")
from paths import DATA_PATH, CODE_PATH

CONFIG_PATH = f"{CODE_PATH}/config"
GAZE_PATH = f"{DATA_PATH}/gaze"
SETS_PATH = f"{DATA_PATH}/sets"
SAMPLES_PATH = f"{DATA_PATH}/samples"
GROUND_TRUTHS_PATH = f"{DATA_PATH}/ground_truths"
MODELS_PATH = f"{DATA_PATH}/models"

SALICON_PATH = f"{DATA_PATH}/salicon"
RAW_SALICON_PATH = f"{SALICON_PATH}/raw"
RAW_SALICON_GAZES_PATH = f"{RAW_SALICON_PATH}/gazes"
RAW_SALICON_IMAGES_PATH = f"{RAW_SALICON_PATH}/images"
PROCESSED_SALICON_PATH = f"{SALICON_PATH}/processed"

DHF1K_PATH = f"{DATA_PATH}/dhf1k"
RAW_DHF1K_PATH = f"{DHF1K_PATH}/raw"
RAW_EXPORTDATA_DHF1K_PATH = f"{RAW_DHF1K_PATH}/exportdata"
RAW_VIDEOS_DHF1K_PATH = f"{RAW_DHF1K_PATH}/videos"
PROCESSED_DHF1K_PATH = f"{DHF1K_PATH}/processed"

FIXATION_DATA_PATH = f"{GAZE_PATH}/fixation_data.csv"

SEQUENCE_LENGTH = 3
BATCH_SIZE = 2
SPLITS = (0.7, 0.2, 0.1)
N_WORKERS = 4

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"