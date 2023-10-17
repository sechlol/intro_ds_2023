import pandas as pd
from . import DataSource


class CsvDataSource(DataSource):

    def __init__(self, path: str):
        self._path = path

    @property
    def name(self) -> str:
        return f"CSV {self._path}"

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self._path, index_col=0, parse_dates=True)
