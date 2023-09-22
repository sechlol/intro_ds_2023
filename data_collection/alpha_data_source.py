from typing import Optional

import pandas as pd

from .data_source import DataSource


class AlphaDataSource(DataSource):
    """
    Get data from AlphaVantage using HTTP requests
    Check https://www.alphavantage.co/documentation/
    """

    # Get api key from https://www.alphavantage.co/support/#api-key
    _API_KEY = "we need an api KEY!"

    @property
    def name(self) -> str:
        return "AlphaVantage"

    def get_data(self) -> Optional[pd.DataFrame]:
        return None
