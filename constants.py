
# Excel file for distributions
DIST_EXCEL_FILE = "./data/distribution_data.xlsx"
AVE_SHEET = "average"
STD_DEV_SHEET = "std_dev"
MIN_SHEET = "min"
MAX_SHEET = "max"
TEST_SHEET = "test_data"

# Excel file for results of nn model
RESULTS_EXCEL_FILE = "./saved_model/results.xlsx"
RESULTS_SHEET = "results_1"

# data properties
NUM_PROPERTIES = 7
NUM_CLASSES = 9

# number of samples per class
SAMPLES_PER_CLASS = 100

# model properties
BATCH_SIZE = 32
INPUT_SIZE = NUM_PROPERTIES
HIDDEN_SIZE = 32
OUTPUT_SIZE = NUM_CLASSES
NUM_EPOCHS = 200
LEARNING_RATE = 0.05
TRAIN_PORTION = 0.8
MODEL_NAME = "EOR_model.pth"
MODEL_PATH = "saved_model"

# PCA properties
PCA_ENABLED = 0
PCA_COMPONENTS = 3
