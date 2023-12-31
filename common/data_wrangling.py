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

    # Standardize relative strength to 0 mean and unit variance
    return (df - df.mean()) / df.std()


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


def compute_returns(dataset: pd.DataFrame, indices: List[str], period_differences: List[int]) -> pd.DataFrame:
    """
    Calculate relative returns on the specified period difference
    """
    subset = dataset[indices]
    returns = []
    for period in period_differences:
        shifted = subset.shift(periods=period)
        ret = (subset - shifted) / shifted
        ret = ret.add_suffix(f"_return{period}").dropna()
        returns.append(ret)

    return pd.concat(returns, axis=1).dropna()
