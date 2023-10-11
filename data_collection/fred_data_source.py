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
    _REAL_TIME_DATE = datetime(2023, 10, 1)

    @property
    def name(self) -> str:
        return "FRED"

    def get_data(self) -> Optional[pd.DataFrame]:
        series_ids = [
            "GDPC1",   # Real Gross Domestic Product
            "UNRATE",   # Unemployment rate
            "INDPRO",    # Industrial Production: Total Index
            "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
            "T10Y3M",   # 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
            "PAYEMS",   # All Employees, Total Nonfarm
            "UMCSENT",  # Consumer Sentiment, University of Michigan
            "UMDMNO",   # Manufacturers' New Orders: Durable Goods
            "FEDFUNDS",  # Federal Funds Effective Rate
            "VIXCLS",   # CBOE Volatility Index: VIX
            "DCOILWTICO",   # Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
            # "WPU10210501",  # Gold Ore: Producer Price Index by Commodity: Metals and Metal Products
            "CPIAUCSL",     # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
            "AWHMAN",  # Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing
            "NEWORDER",  # Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
            "ACOGNO",  # Manufacturers' New Orders: Consumer Goods
            "CES4348400001",    # All Employees, Truck Transportation
            #"USSLIND",  # The leading index for each state predicts the six-month growth rate of the state's coincident index. In addition to the coincident index, variables that lead the economy: state-level housing permits (1 to 4 units), state initial unemployment insurance claims, delivery times from the Institute for Supply Management (ISM) manufacturing survey, and the interest rate spread between the 10-year Treasury bond and the 3-month Treasury bill.
            #"AWHAETP",  # Average weekly hours of All Employees, Total Private
            #"M08297USM548NNBR" # Initial Claims, Unemployment Insurance, State Programs for United States
        ]

        data_frames = []
        for series_id in series_ids:
            df = self._get_dfs(series_id)
            data_frames.append(df)

        complete_df = pd.concat(data_frames, axis=1).dropna()
        return complete_df

    def _make_url(self, series_id: str):
        parameters = {
            "series_id": series_id,
            #"realtime_start": self._REAL_TIME_DATE.strftime("%Y-%m-%d")
        }

        url = self._URL
        for par, val in parameters.items():
            url += f"&{par}={val}"

        return url

    def _get_dfs(self, series_id):

        print("\t- ", series_id)
        url = self._make_url(series_id)
        raw_data = requests.get(url)
        json_data = json.loads(raw_data.text)
        observations = json_data["observations"]
        df = pd.DataFrame(observations)

        filtered_df = df.loc[:, ["date", "value"]]
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        filtered_df['value'] = pd.to_numeric(filtered_df['value'], errors="coerce")

        # Handle duplicate dates by grouping and keeping mean value
        filtered_df = (filtered_df
                       .sort_values(by='date')
                       .groupby('date')['value']
                       .mean()  # Keep the mean value for each duplicate date
                       .reset_index())

        # Create a new date range that goes from _START_DATE to today
        # This will make sure that missing data for recent days will be forward filled from the latest available reading
        date_range = pd.date_range(start=FredDataSource._START_DATE, end=datetime.now(), freq='B')

        return (filtered_df[filtered_df["date"] >= FredDataSource._START_DATE]
                .dropna()
                .rename(columns={"value": series_id})
                .set_index('date')
                .resample("B").ffill()
                .reindex(date_range).ffill())
