from typing import Optional

import pandas as pd

from .data_source import DataSource


class YahooDataSource(DataSource):
    """
    Uses yfinance library to fetch data from YAHOO
    Check https://github.com/ranaroussi/yfinance
    """
    @property
    def name(self) -> str:
        return "Yahoo!"

    def get_data(self) -> Optional[pd.DataFrame]:
        return None
