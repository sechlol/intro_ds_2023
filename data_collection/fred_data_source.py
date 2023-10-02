from typing import Optional

import pandas as pd

from .data_source import DataSource


class FredDataSource(DataSource):
    """
    Get data from Fred using fredapi
    Check https://github.com/mortada/fredapi
    """

    @property
    def name(self) -> str:
        return "HelloHello"

    def get_data(self) -> Optional[pd.DataFrame]:
        return None
