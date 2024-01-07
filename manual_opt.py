"""Contains functions to run ANN model based on parameter input from an Excel file

Excel file name is defined in constants.py
Hyperparameters are read from the Excel file and train, validation, and test accuracies
are written to the same Excel file after all calculations are finished.
"""

import pandas as pd
from typing import List, Tuple
from model_utils import load_data, run_model
from constants import RESULTS_EXCEL_FILE, RESULTS_SHEET


def read_results_file() -> Tuple[List[List[float]], List[str]]:
    """Reads data from an Excel file

    Each row in the Excel file contains a set of hyperparameters

    :return:
        a 2d list of hyperparameters to be tested
        a 1d list of hyperparameter names
    """
    params_df = pd.read_excel(RESULTS_EXCEL_FILE, sheet_name=RESULTS_SHEET)
    params_df.fillna(0, inplace=True)
    return params_df.values.tolist(), params_df.columns.tolist()


def save_results(data: List[List[float]], labels: List[str]) -> None:
    """Saves a 2d list to an Excel file

    Uses labels list as the headers

    :param data: a 2d list of floats to be saved in the Excel file
    :param labels: a 1d list representing column headers
    :return:
        None
    """
    params_df = pd.DataFrame(data=data, columns=labels)
    with pd.ExcelWriter(RESULTS_EXCEL_FILE) as writer:
        params_df.to_excel(excel_writer=writer, sheet_name=RESULTS_SHEET, float_format="%.3f",
                           index=False, header=True)


def run_multiple() -> None:
    """Runs an ANN model multiple times based on hyperparameters read from an Excel file

    :return:
        None
    """
    input_dataset, test_dataset = load_data()
    # read model parameters to be tested
    param_list, param_labels = read_results_file()
    # perform the train and test steps once per each set of parameters ##
    for i, params in enumerate(param_list):
        # set hyper-parameters
        hyper_params = {
            'batch_size': params[0],
            'hidden_size': params[1],
            'learning_rate': params[2],
            'train_portion': params[3],
            'samples_per_class': params[4],
            'num_epochs': params[5],
        }
        params[-3:] = run_model(input_dataset=input_dataset, test_dataset=test_dataset, params=hyper_params)
    # save results to the Excel file
    save_results(data=param_list, labels=param_labels)


if __name__ == "__main__":
    run_multiple()
