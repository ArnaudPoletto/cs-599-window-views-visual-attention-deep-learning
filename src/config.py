import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

DATA_PATH = str(GLOBAL_DIR / "data")
GAZE_PATH = f"{DATA_PATH}/gaze"
SETS_PATH = f"{DATA_PATH}/sets"
SAMPLES_PATH = f"{DATA_PATH}/samples"

FIXATION_DATA_PATH = f"{GAZE_PATH}/fixation_data.csv"