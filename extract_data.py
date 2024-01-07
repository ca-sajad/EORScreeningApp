"""Contains the function to extract oilfield properties data
"""


import pandas as pd
from typing import List, Tuple
from constants import *


def read_excel_file() -> Tuple[List[List[str | float]], List[List[str | float]], List[List[str | float]]]:
    """Reads oilfield properties from Excel files

    Retrieves minimum and maximum of oilfield properties for all EOR methods, and
    retrieves oilfield data to serve as test dataset

    :arg:
        None

    :returns:
        a list of minimum values of properties for each method,
            e.g. [['chemical', 0.5, 2.0, 10.0], ['co2_miscible', 0.3, 1.1, 14.0]]
        a list of maximum values of properties for each method,
            e.g. [['chemical', 2.5, 4.0, 30.0], ['co2_miscible', 0.9, 2.3, 40.0]]
        a list of oilfield properties,
            e.g. [['chemical', 0.6, 2.13, 15.6], ['chemical', 1.9, 2.3, 22.0]]
    """
    # reading properties' "min" data
    min_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=MIN_SHEET)
    min_list = min_df.values.tolist()

    # reading properties' "max" data
    max_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=MAX_SHEET)
    max_list = max_df.values.tolist()

    # reading properties' "test" data
    test_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=TEST_SHEET)
    test_list = test_df.values.tolist()

    return min_list, max_list, test_list
