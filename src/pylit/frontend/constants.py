from pathlib import Path

PATH_PROJECT = str(Path(__file__).parent.parent.parent.parent)
PATH_SETTINGS = PATH_PROJECT + "/app_settings.json"
SCI_NUM_STEP = 1e-15
NUM_STEP = 1
DEFAULT_INT = 0
DEFAULT_FLOAT = 0.0
DEFAULT_LOWER_VALUE = 0.0
DEFAULT_UPPER_VALUE = 1.0
DEFAULT_NUM_VALUE = 10