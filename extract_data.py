"""Contains functions to extract oilfield properties data
"""


import pandas as pd
from typing import List, Tuple, Dict
from constants import *


def read_distribution_file() -> Dict[str, List[List[str | float] | str]]:
    """Reads min and max of oilfield properties from an Excel file

    Retrieves minimum and maximum of oilfield properties for all EOR methods, and
    returns a dictionary of these data.

    :arg:
        None

    :returns:
    a dictionary containing
        - key: min_list, value: a 2d list where each element contains minimum values of properties for an EOR method,
            e.g. [['chemical', 0.5, 2.0, 10.0], ['co2_miscible', 0.3, 1.1, 14.0]]
        - key: max_list, value: a 2d list where each element contains maximum values of properties for an EOR method,
            e.g. [['chemical', 2.5, 4.0, 30.0], ['co2_miscible', 0.9, 2.3, 40.0]]
        - key: prop_labels, value: a 1d list of property names
    """

    data_dict = {}

    min_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=MIN_SHEET)
    data_dict['min_list'] = min_df.values.tolist()
    data_dict['prop_labels'] = min_df.columns.tolist()[1:]

    max_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=MAX_SHEET)
    data_dict['max_list'] = max_df.values.tolist()

    return data_dict


def read_test_dataset_file() -> Dict[str, List[List[float] | str]]:
    """Reads oilfield properties of test dataset from an Excel file

    Retrieves oilfield data to serve as test dataset.

    :arg:
        None

    :returns:
        a 2d list of oilfield properties,
            e.g. [[0.6, 2.13, 15.6], [1.9, 2.3, 22.0]]
        a 1d list of EOR methods corresponding to elements in test_data

    """
    test_dict = {}

    test_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=TEST_SHEET)
    test_list = test_df.values.tolist()

    test_dict['sample_data'] = [sample[1:] for sample in test_list]
    test_dict['sample_label'] = [sample[0] for sample in test_list]
    test_dict['prop_labels'] = test_df.columns.tolist()[1:]

    return test_dict
