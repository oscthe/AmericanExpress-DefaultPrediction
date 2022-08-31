# Path-variables
INPUT_FOLDER = "../input/"
RAW_DATA = INPUT_FOLDER + "train_data.ftr"
TRAINING_LABELS = INPUT_FOLDER + "train_labels.csv"
TEST_DATA = INPUT_FOLDER + "test_data.ftr"

TRANSFORMED_DATA = INPUT_FOLDER + "transformed_train.csv"
TRAIN_DATA = INPUT_FOLDER + "train_folds.csv"

MODEL_OUTPUT = "../models/"

# Constants
SEED = 42
CAT_FEATURES = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
