from typing import Optional
import requests
import pandas as pd
from datetime import datetime

from .data_source import DataSource



print(date_vals)
    @property
    def name(self) -> str:
        return "Yahoo!"

    class FredDataSource(DataSource):
        """
        Get data from Fred using fredapi
        Check https://github.com/mortada/fredapi
        """
        _API_KEY = "86d47b19c7da7f5043b731d2d982755c"
        _URL = f"https://api.stlouisfed.org/fred/series/observations?series_id={ticker}&realtime_start=&api_key={_API_KEY}&file_type=json"
        _START_DATE = datetime(1970, 1, 1)

        @property
        def name(self) -> str:
            return "FRED"


    def get_data(self) -> Optional[pd.DataFrame]:
        dfs = [
            self._get_ticker("USSLIND"), # The leading index for each state predicts the six-month growth rate of the state's coincident index. In addition to the coincident index, variables that lead the economy: state-level housing permits (1 to 4 units), state initial unemployment insurance claims, delivery times from the Institute for Supply Management (ISM) manufacturing survey, and the interest rate spread between the 10-year Treasury bond and the 3-month Treasury bill.
            self._get_ticker("T10Y2Y"), # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
            self._get_ticker("AWHAETP"), # Average weekly hours of All Employees, Total Private
            self._get_ticker("AWHMAN"), # Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing
            self._get_ticker("NEWORDER"), # Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
            self._get_ticker("ACOGNO"), # Manufacturers' New Orders: Consumer Goods
            self._get_ticker("M08297USM548NNBR"), # Initial Claims, Unemployment Insurance, State Programs for United States
        ]
        response = requests.get(_URL)
        data = response.json()
        df = pd.DataFrame(data['observations'])
        date_vals = date_vals = df[['date', 'value']]
        return pd.concat(dfs, axis=1).dropna()


    def _make_url(self, function: str, interval: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        params["function"] = function
        if interval:
            params["interval"] = interval

        return self._URL + "&" + "&".join([f"{name}={val}" for name, val in params.items()])


    def _read_df(url: str, resample_frequency: Optional[str] = "B"):
        """
        :param url: The URL to read CSV from
        :param resample_frequency: Transforms the original reading frequency to the frequency specified. For a list
        of available frequencies, check https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        """
        print(url)
        df = pd.read_csv(url, parse_dates=True, index_col="timestamp")
        return (df[df.index >= AlphaDataSource._OLDEST_DATE_LIMIT]
                .resample(resample_frequency)
                .ffill())



    def _get_ticker(self, ticker: str):
        """
        Get data between the given time interval from FRED.
        """
        print(f"\t- {ticker}")
        df = yf.Ticker(ticker).history(start=self._START_DATE)
        return (df
                .rename(columns={"Close": ticker})     # Rename "Close" to the ticker name
                .loc[:, ticker]                        # Select only the Closing price
                .to_frame()                         # Transform Series to Dataframe
                .tz_localize(None)                  # Remove timezone info form index
                .resample('B')                      # Resample to Business Days
                .ffill())                           # Forward fill data
