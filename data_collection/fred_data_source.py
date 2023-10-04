import json
from typing import Optional
import requests
import pandas as pd
from datetime import datetime

from .data_source import DataSource


class FredDataSource(DataSource):
    """
    Get data from Fred using fredapi
    Check https://github.com/mortada/fredapi
    """
    _API_KEY = "86d47b19c7da7f5043b731d2d982755c"
    _URL = f"https://api.stlouisfed.org/fred/series/observations?api_key={_API_KEY}&file_type=json"
    _START_DATE = datetime(1970, 1, 1)

    @property
    def name(self) -> str:
        return "FRED"

    def get_data(self) -> Optional[pd.DataFrame]:
        dfs = [
            self._get_info("USSLIND", ),
            # The leadinfodex for each state predicts the six-month growth rate of the state's coincident index. In addition to the coincident index, variables that lead the economy: state-level housing permits (1 to 4 units), state initial unemployment insurance claims, delivery times from the Institute for Supply Management (ISM) manufacturing survey, and the interest rate spread between the 10-year Treasury bond and the 3-month Treasury bill.
            # self._get_info("T10Y2Y"),  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
            # self._get_info("AWHAETP"),  # Average weekly hours of All Employees, Total Private
            # self._get_info("AWHMAN"),  # Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing
            # self._get_info("NEWORDER"),  # Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
            # self._get_info("ACOGNO"),  # Manufacturers' New Orders: Consumer Goods
            # self._get_info("M08297USM548NNBR"),
            # Initial Claims, Unemployment Insurance, State Programs for United States
        ]
        return pd.concat(dfs, axis=1).dropna()

    def _make_url(self, series_id: str):
        parameters = {
            "series_id": series_id,
            "realtime_start": self._START_DATE.strftime("%Y-%m-%d")
        }

        url = self._URL
        for par, val in parameters.items():
            url += f"&{par}={val}"

        return url

    def _get_info(self, series_id):
        url = self._make_url(series_id)
        raw_data = requests.get(url)
        json_data = json.loads(raw_data.text)
        observations = json_data["observations"]
        df = pd.DataFrame(observations)
        filtered_df = df[["date", "value"]]
        lel = 0
