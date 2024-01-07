"""Contains constant values for file names and model properties

These constant values include Excel file paths and sheet names, and
hyperparameters used to create ANN model. Hyperparameters may be
overwritten in the course of model optimization, but the ones in this
file are the optimized hyperparameters.
"""


# Excel file for distributions
DIST_EXCEL_FILE = "./data/distribution_data.xlsx"
AVE_SHEET = "average"       # is not used
STD_DEV_SHEET = "std_dev"   # is not used
MIN_SHEET = "min"
MAX_SHEET = "max"
TEST_SHEET = "test_data"

# Excel file for results of nn model
RESULTS_EXCEL_FILE = "./manual_opt_results/results.xlsx"
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

