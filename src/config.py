import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch

DATA_PATH = str(GLOBAL_DIR / "data")
GAZE_PATH = f"{DATA_PATH}/gaze"
SETS_PATH = f"{DATA_PATH}/sets"
SAMPLES_PATH = f"{DATA_PATH}/samples"
GROUND_TRUTHS_PATH = f"{DATA_PATH}/ground_truths"
MODELS_PATH = f"{DATA_PATH}/models"
SALICON_PATH = f"{DATA_PATH}/salicon"
RAW_SALICON_PATH = f"{SALICON_PATH}/raw"
PROCESSED_SALICON_PATH = f"{SALICON_PATH}/processed"
RAW_SALICON_GAZES_PATH = f"{RAW_SALICON_PATH}/gazes"
RAW_SALICON_IMAGES_PATH = f"{RAW_SALICON_PATH}/images"
FIXATION_DATA_PATH = f"{GAZE_PATH}/fixation_data.csv"

SEQUENCE_LENGTH = 3
WITH_TRANSFORMS = True
BATCH_SIZE = 2
SPLITS = (0.7, 0.2, 0.1)
N_WORKERS = 0

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"