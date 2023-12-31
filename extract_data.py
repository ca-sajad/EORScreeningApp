
import pandas as pd
from typing import List, Tuple

DIST_EXCEL_FILE = "./data/distribution_data.xlsx"
AVE_SHEET = "average"
STD_DEV_SHEET = "std_dev"


def extract() -> Tuple[List[List[str | float]], List[List[str | float]]]:
    """
    reads standard deviation and average of property distributions for EOR projects from an Excel file
    :return: list of average values of properties, list of sta dev values of properties
    """
    # reading properties' "average" data
    ave_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=AVE_SHEET)
    ave_list = ave_df.values.tolist()

    # reading properties' "standard deviation" data
    std_dev_df = pd.read_excel(DIST_EXCEL_FILE, sheet_name=STD_DEV_SHEET)
    std_dev_list = std_dev_df.values.tolist()

    return ave_list, std_dev_list
