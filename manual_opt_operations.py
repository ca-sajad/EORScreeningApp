
import pandas as pd
from typing import List, Tuple
from constants import *


def read_results_file() -> Tuple[List[List[float]], List[str]]:
    params_df = pd.read_excel(RESULTS_EXCEL_FILE, sheet_name=RESULTS_SHEET)
    params_df.fillna(0, inplace=True)
    return params_df.values.tolist(), params_df.columns.tolist()


def save_results(data: List[List[float]], labels: List[str]) -> None:
    params_df = pd.DataFrame(data=data, columns=labels)
    with pd.ExcelWriter(RESULTS_EXCEL_FILE) as writer:
        params_df.to_excel(excel_writer=writer, sheet_name=RESULTS_SHEET, float_format="%.3f",
                           index=False, header=True)
