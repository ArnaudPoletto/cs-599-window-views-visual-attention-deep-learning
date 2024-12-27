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
MODELS_PATH = f"{DATA_PATH}/models"
CHECKPOINTS_PATH = f"{DATA_PATH}/checkpoints"
SETS_PATH = f"{DATA_PATH}/sets"

SALICON_PATH = f"{DATA_PATH}/salicon"
RAW_SALICON_PATH = f"{SALICON_PATH}/raw"
RAW_SALICON_GAZES_PATH = f"{RAW_SALICON_PATH}/gazes"
RAW_SALICON_IMAGES_PATH = f"{RAW_SALICON_PATH}/images"
RAW_SALICON_GROUND_TRUTHS_PATH = f"{RAW_SALICON_PATH}/ground_truths"
TEST_SALICON_PATH = f"{SALICON_PATH}/test"
PROCESSED_SALICON_PATH = f"{SALICON_PATH}/processed"

DHF1K_PATH = f"{DATA_PATH}/dhf1k"
RAW_DHF1K_PATH = f"{DHF1K_PATH}/raw"
RAW_EXPORTDATA_DHF1K_PATH = f"{RAW_DHF1K_PATH}/exportdata"
RAW_VIDEOS_DHF1K_PATH = f"{RAW_DHF1K_PATH}/videos"
PROCESSED_DHF1K_PATH = f"{DHF1K_PATH}/processed"

VIEWOUT_PATH = f"{DATA_PATH}/viewout"
RAW_VIEWOUT_PATH = f"{VIEWOUT_PATH}/raw"
PROCESSED_VIEWOUT_PATH = f"{VIEWOUT_PATH}/processed"

IMAGE_ENCODER_N_LEVELS = 5
IMAGE_ENCODER_MODEL_NAME = "pnasnet5large"
IMAGE_ENCODER_PRETRAINED = True
SIMPLE_NET_N_LEVELS = 5
SIMPLE_NET_MODEL_NAME = "semnasnet_100"
SIMPLE_NET_PRETRAINED = True

LOSS_WEIGHTS = {
    "kl": 1.0,
    "cc": 1.0,
}

IMAGE_SIZE = 331
FINAL_HEIGHT = 480
FINAL_WIDTH = 640
VIEWOUT_FRAME_HEIGHT = 3072
VIEWOUT_FRAME_WIDTH = 6144
SEQUENCE_LENGTH = 5
N_WORKERS = 4

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"