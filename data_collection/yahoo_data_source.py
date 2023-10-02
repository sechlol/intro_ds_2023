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

    @staticmethod
    def interval():
        start = datetime(1970, 1, 1)
        end = datetime(2023, 10, 1)
        return [start, end]

    @property
    def name(self) -> str:
        return "Yahoo!"

    def get_data(self) -> Optional[pd.DataFrame]:
        dfs = [
            self._get_xle(),
            self._get_xly(),
            self._get_xlf(),
            self._get_xlv(),
            self._get_xli(),
            self._get_xlk(),
            self._get_xlb(),
            self._get_xlre(),
            self._get_xlc(),
            self._get_xlu(),
            self._get_xlp()
        ]
        return pd.concat(dfs, axis=1).dropna()

    def _read_df(self, ind: str):
        """
        Get data between the given time interval from yfinance.
        :param ind: One of the following indices:
        ['XLE', 'XLY', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLC', 'XLU', 'XLP']
        """
        df = yf.Ticker(ind).history(start=self.interval()[0], end=self.interval()[1])
        df = df.rename(columns={"Close": ind}).resample('B').ffill()
        df.index = df.index.date
        return df[ind]

    def _get_xle(self, ind: str = "XLE") -> pd.DataFrame:
        """
        Get XLE data
        :param ind: XLE
        """
        return self._read_df(ind)

    def _get_xly(self, ind: str = "XLY") -> pd.DataFrame:
        """
        Get XLY data
        :param ind: XLY
        """
        return self._read_df(ind)

    def _get_xlf(self, ind: str = "XLF") -> pd.DataFrame:
        """
        Get XLF data
        :param ind: XLF
        """
        return self._read_df(ind)

    def _get_xlv(self, ind: str = "XLV") -> pd.DataFrame:
        """
        Get XLV data
        :param ind: XLV
        """
        return self._read_df(ind)

    def _get_xli(self, ind: str = "XLI") -> pd.DataFrame:
        """
        Get XLI data
        :param ind: XLI
        """
        return self._read_df(ind)

    def _get_xlk(self, ind: str = "XLK") -> pd.DataFrame:
        """
        Get XLK data
        :param ind: XLK
        """
        return self._read_df(ind)

    def _get_xlb(self, ind: str = "XLB") -> pd.DataFrame:
        """
        Get XLB data
        :param ind: XLb
        """
        return self._read_df(ind)

    def _get_xlre(self, ind: str = "XLRE") -> pd.DataFrame:
        """
        Get XLRE data
        :param ind: XLRE
        """
        return self._read_df(ind)

    def _get_xlc(self, ind: str = "XLC") -> pd.DataFrame:
        """
        Get XLC data
        :param ind: XLC
        """
        return self._read_df(ind)

    def _get_xlu(self, ind: str = "XLU") -> pd.DataFrame:
        """
        Get XLU data
        :param ind: XLU
        """
        return self._read_df(ind)

    def _get_xlp(self, ind: str = "XLP") -> pd.DataFrame:
        """
        Get XLP data
        :param ind: XLP
        """
        return self._read_df(ind)
