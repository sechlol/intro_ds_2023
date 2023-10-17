from itertools import combinations
from typing import List

import pandas as pd


def compute_relative_strength(dataset: pd.DataFrame, indices: List[str]) -> pd.DataFrame:
    """
        Calculate relative strengths between stock indices. The algorithm will choose subsequent
        combinations of indices with no repetitions. See combinations() documentations for details:
        https://docs.python.org/3/library/itertools.html#itertools.combinations

        Args:
            dataset (pd.DataFrame): DataFrame with historical data for stock indices.
            indices (List[str]): List of index names to compare.

        Returns:
            pd.DataFrame: DataFrame containing relative strengths between index pairs, normalized
            between 0 and 1. The resulting DataFrame will have column labels in the form of "NUMERATOR/DENOMINATOR"
        """
    relative_strengths = {}

    # Calculate relative strength for all combinations of indices
    for numerator, denominator in combinations(indices, 2):
        tag = f"{numerator}/{denominator}"
        relative_strengths[tag] = dataset[numerator] / dataset[denominator]

    # Create a DataFrame from the relative strength dictionary
    df = pd.DataFrame(relative_strengths)

    # Normalize relative strength values to the range [0, 1]
    return (df - df.min()) / (df.max() - df.min())


def compute_SMA(dataset: pd.DataFrame, indices: List[str], window_lengths: List[int]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) for stock indices with different window lengths.

    Args:
        dataset (pd.DataFrame): Historical data for stock indices.
        indices (List[str]): List of index names for SMA calculation.
        window_lengths (List[int]): List of window lengths for SMA calculation.

    Returns:
        pd.DataFrame: DataFrame containing SMA values for each index with respective window lengths.
    """
    rolled_datasets = [
        (dataset[indices]
         .rolling(window=length)
         .mean()
         .add_suffix(f"_SMA{length}"))
        for length in window_lengths]

    return pd.concat(rolled_datasets, axis=1).dropna()

def momentum(dataset: pd.DataFrame, indices: List[str], interval: int) -> pd.DataFrame:
    """
        Compute momentum, that is rate of change in the returns of an index
    """
    momentum = dataset[indices].copy()
    for col_name in indices:
        momentum = (dataset - dataset.shift(interval)) / dataset.shift(interval) * 100
    momentum.fillna(method='bfill', inplace=True)
    momentum = momentum.add_suffix(f"_MOM")
    return momentum



from sklearn.preprocessing import MinMaxScaler


def diy_lei(dataset: pd.DataFrame) -> pd.DataFrame:
    """"
         Calculate an idex of leading indicators, mimicking the Conference Board LEI: https://www.conference-board.org/topics/business-cycle-indicators/press/us-lei-nov-2021
         Calculate our own LEI:
         sum all the indicators, normalize, for each data point in time:
    """

    leading_indicators = ['SPY', 'T10Y2Y', 'T10Y3M', 'PAYEMS', 'ICSA', 'UMCSENT', 'UMDMNO', 'PPIACO', 'AWHMAN',
                          'NEWORDER', 'ACOGNO']

    data_scaled = pd.DataFrame(index=dataset.index)

    scaler = MinMaxScaler()  # default=(0, 1)

    for indicator in leading_indicators:
        data_scaled[indicator] = scaler.fit_transform(dataset[[indicator]]).flatten()

    # Creating a composite indicator as a sum of all scaled indicators
    lei_values = data_scaled.sum(axis=1)
    diy_lei = pd.DataFrame({
        'DIY LEI': lei_values
    })
    return diy_lei


def diy_lag(dataset: pd.DataFrame) -> pd.DataFrame:
    """"
         Calculate an index of lagging indicators, loosely inspired by the Conference Board LAG: https://www.conference-board.org/topics/business-cycle-indicators/press/us-lei-nov-2021
         Maybe we have to average, think about that. Not sure.
    """

    lagging_indicators = ['GDPC1', 'UNRATE', 'FEDFUNDS', 'CPIAUCSL']
    data_scaled = pd.DataFrame(index=dataset.index)

    scaler = MinMaxScaler()  # default=(0, 1)

    for indicator in lagging_indicators:
        data_scaled[indicator] = scaler.fit_transform(dataset[[indicator]]).flatten()

    # Creating a composite indicator as a sum of all scaled indicators
    lag_values = data_scaled.sum(axis=1)
    diy_lag = pd.DataFrame({
        'DIY LAG': lag_values
    })
    return diy_lag

