from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from .data_source import DataSource


class AlphaDataSource(DataSource):
    """
    Get data from AlphaVantage using HTTP requests
    Check https://www.alphavantage.co/documentation/
    """

    # Get api key from https://www.alphavantage.co/support/#api-key
    _API_KEY = "CIVPZPUWEE594XT1"
    _URL = f"https://www.alphavantage.co/query?apikey={_API_KEY}&datatype=csv"
    _OLDEST_DATE_LIMIT = datetime(1970, 1, 1)

    @property
    def name(self) -> str:
        return "AlphaVantage"

    def get_data(self) -> Optional[pd.DataFrame]:
        dfs = [
            self._get_technical("RSI", "SPY"),
            self._get_technical("PPO", "SPY"),
            self._get_technical("MOM", "SPY"),
            self._get_technical("ROCR", "SPY"),
            self._get_technical("ULTOSC", "SPY"),
        ]
        return pd.concat(dfs, axis=1).dropna()

    def _make_url(self, function: str, interval: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        params["function"] = function
        if interval:
            params["interval"] = interval

        return self._URL + "&" + "&".join([f"{name}={val}" for name, val in params.items()])

    @staticmethod
    def _read_df(url: str, resample_frequency: Optional[str] = "B") -> pd.DataFrame:
        """
        :param url: The URL to read CSV from
        :param resample_frequency: Transforms the original reading frequency to the frequency specified. For a list
        of available frequencies, check https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        """
        print(f"\t- {url}")
        df = pd.read_csv(url, parse_dates=True, index_col=0, date_format="%Y-%m-%d")
        return (df[df.index >= AlphaDataSource._OLDEST_DATE_LIMIT]
                .resample(resample_frequency)
                .ffill())

    @staticmethod
    def _read_df_macro(url: str, rename_value: str, resample_frequency: Optional[str] = "B"):
        """
        In addition to _read_df, also renames the "value" field to the specified "rename_value"
        """
        return (AlphaDataSource
                ._read_df(url, resample_frequency)
                .rename(columns={"value": rename_value}))

    @staticmethod
    def _read_df_ticker(url: str, rename_value: str, resample_frequency: Optional[str] = "B"):
        """
        In addition to _read_df, keeps only the "close" column and renames it
        """
        df = AlphaDataSource._read_df(url, resample_frequency)
        return df["close"].rename(rename_value)

    @staticmethod
    def _read_df_technical(url: str, rename_value: str, resample_frequency: Optional[str] = "B"):
        """
        In addition to _read_df, also renames the first column field to the specified "rename_value"
        """
        df = AlphaDataSource._read_df(url, resample_frequency)
        df.rename(columns={df.columns[0]: rename_value}, inplace=True)
        return df

    def _get_ticker(self, ticker: str) -> pd.DataFrame:
        """
        By default, interval=annual. Strings quarterly and annual are accepted.
        """
        url = self._make_url("TIME_SERIES_DAILY", params={"symbol": ticker, "outputsize": "full"})
        df = AlphaDataSource._read_df(url).rename(columns={"close": ticker})
        return df[ticker].to_frame()

    def _get_treasury_bond_yield(self, maturity: str = "10year") -> pd.DataFrame:
        """
        By default, interval=monthly. Strings daily, weekly, and monthly are accepted.
        :param maturity: By default, maturity=10year. Accepted: 3month, 2year, 5year, 7year, 10year, 30year
        """
        url = self._make_url("TREASURY_YIELD", "monthly", {"maturity": maturity})
        return self._read_df_macro(url, maturity+"_yield")

    def _get_gdp(self) -> pd.DataFrame:
        """
        By default, interval=annual. Strings quarterly and annual are accepted.
        """
        url = self._make_url("REAL_GDP", "quarterly")
        return self._read_df_macro(url, "gdp")

    def _get_cpi(self) -> pd.DataFrame:
        """
        By default, interval=monthly. Strings monthly and semiannual are accepted.
        """
        url = self._make_url("CPI", "monthly")
        return self._read_df_macro(url, "cpi")

    def _get_unemployment(self) -> pd.DataFrame:
        """
        This API returns the monthly unemployment data of the United States.
        The unemployment rate represents the number of unemployed as a percentage of the labor force.
        Labor force data are restricted to people 16 years of age and older, who currently reside in 1 of the 50 states
        or the District of Columbia, who do not reside in institutions (e.g., penal and mental facilities,
        homes for the aged), and who are not on active duty in the Armed Forces
        """
        url = self._make_url("UNEMPLOYMENT")
        return self._read_df_macro(url, "unemployment")

    def _get_inflation(self):
        url = self._make_url("INFLATION")
        return self._read_df_macro(url, "inflation")

    def _get_technical(self, function: str, symbol: str):
        url = self._make_url(function, interval="daily", params={
            "symbol": symbol,
            "series_type": "close",
            "time_period": 60
        })
        return self._read_df_technical(url, rename_value=f"{symbol}_{function}")
