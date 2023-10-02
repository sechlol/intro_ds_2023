from typing import Optional

import pandas as pd
import yfinance as yf
from datetime import datetime

from .data_source import DataSource


class YahooDataSource(DataSource):
    """
    Uses yfinance library to fetch data from YAHOO
    Check https://github.com/ranaroussi/yfinance
    """
    _START_DATE = datetime(1970, 1, 1)

    @property
    def name(self) -> str:
        return "Yahoo!"

    def get_data(self) -> Optional[pd.DataFrame]:
        dfs = [
            self._get_ticker("SPY"),
            self._get_ticker("XLE"),
            self._get_ticker("XLY"),
            self._get_ticker("XLF"),
            self._get_ticker("XLV"),
            self._get_ticker("XLI"),
            self._get_ticker("XLK"),
            self._get_ticker("XLB"),
            self._get_ticker("XLU"),
            self._get_ticker("XLP"),
            #self._get_ticker("XLRE"),  # Has data only from 2015
            #self._get_ticker("XLC"),   # Has data only from 2018
        ]
        return pd.concat(dfs, axis=1).dropna()

    def _get_ticker(self, ticker: str):
        """
        Get data between the given time interval from yfinance.
        :param ticker: One of the following indices:
        ['XLE', 'XLY', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLC', 'XLU', 'XLP']
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
