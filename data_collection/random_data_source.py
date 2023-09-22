from typing import Optional, List

import numpy as np
import pandas as pd

from . import DataSource
from datetime import datetime


class RandomDataSource(DataSource):

    def __init__(self,
                 symbols: List[str],
                 start_date: str = '01/01/2020',
                 end_date: str = '12/12/2023'):
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date

    @property
    def name(self) -> str:
        return f"Random Source"

    def get_data(self) -> pd.DataFrame:
        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(self._start_date, "%d/%m/%Y")
        end_date = datetime.strptime(self._end_date, "%d/%m/%Y")

        # Generate business days date range using BDay offset
        days = pd.bdate_range(start_date, end_date)

        # Generate random prices
        data = {s: np.random.uniform(low=50, high=150, size=len(days)) for s in self._symbols}

        # Create a pandas DataFrame
        return pd.DataFrame({"day": days, **data}, index=days)
