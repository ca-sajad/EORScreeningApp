
import pandas as pd
from typing import List, Tuple
from constants import *


def read_excel_file() -> Tuple[List[List[str | float]], List[List[str | float]], List[List[str | float]]]:
    """
    reads characteristics of property distributions of EOR projects from an Excel file
    :return: list of average values of properties, list of sta dev values of properties
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
