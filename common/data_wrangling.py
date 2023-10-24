from itertools import combinations
from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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



def compute_SMA_same_shape(dataset: pd.DataFrame, indices: List[str], window_lengths: List[int]) -> pd.DataFrame:
    # ATTENTION! THIS returns array with same shape
    rolled_datasets = [
        (dataset[indices]
         .rolling(window=length)
         .mean()
         .fillna(dataset[indices])  # Fill NA values with original data
         .add_suffix(f"_SMA{length}"))
        for length in window_lengths]

    return pd.concat(rolled_datasets, axis=1)


def momentum(dataset: pd.DataFrame, indices: List[str], interval: int) -> pd.DataFrame:
    """
        Compute momentum, that is rate of change in the returns of an index
    """

    momentum = (dataset / dataset.shift(interval)) -1
    momentum.fillna(method='bfill', inplace=True)
    #momentum.fillna(0, inplace=True)
    momentum = momentum.add_suffix(f"_MOM")
    return momentum

# List of leading and lagging indicators
LEI_LAG = {
    'leading_indicators': ['SPY', 'T10Y2Y', 'T10Y3M', 'PAYEMS', 'ICSA', 'UMCSENT', 'UMDMNO', 'PPIACO', 'AWHMAN',
                           'NEWORDER', 'ACOGNO'],
    'lagging_indicators': ['GDPC1', 'UNRATE', 'FEDFUNDS', 'CPIAUCSL']
}
def diy_ind(dataset: pd.DataFrame, indices: List[str], name: str) -> pd.DataFrame:
    """"
         Calculate a compound index of leading or lagging indicators, mimicking the Conference Board LEI: https://www.conference-board.org/topics/business-cycle-indicators/press/us-lei-nov-2021
         Calculate our own LEI:/LAG
         sum all the indicators, normalize, for each data point in time.
    """
    data_scaled = pd.DataFrame(index=dataset.index)

    scaler = MinMaxScaler()  # default=(0, 1)

    for indicator in indices:
        data_scaled[indicator] = scaler.fit_transform(dataset[[indicator]]).flatten()
    # Creating a composite indicator as a sum of all scaled indicators
    ind_values = data_scaled.sum(axis = 1)
    diy_indicator = pd.DataFrame({
        f'{name}': ind_values
    })
    return diy_indicator

def aggregate_calcs(dataset: pd.DataFrame) -> pd.DataFrame:
    """""
     Not sure if necessary, but now it's here.
     Make a data frame with the computed indicators.
     Simple Moving Average, Momentum, DIY LEI, DIY LAG
    """""
    all_indices = dataset.columns.tolist()

    # Calculating indicators
    sma_df = compute_SMA_same_shape(dataset, all_indices, [50])
    momentum_df = momentum(dataset, all_indices, 360)
    diy_lei_df = diy_ind(dataset, LEI_LAG['leading_indicators'], 'LEI')
    diy_lag_df = diy_ind(dataset, LEI_LAG['lagging_indicators'], 'LAG')

    # Concatenating all calculated DataFrames along columns
    joint_data = pd.concat([sma_df, momentum_df, diy_lei_df, diy_lag_df], axis=1)

    joint_data_clean = joint_data.dropna()

    return joint_data_clean


def all_data(dataset) -> pd.DataFrame:
    df_indicators = aggregate_calcs(dataset)
    all_indices = dataset.columns.tolist()
    indices_inds = df_indicators.columns.tolist()
    forward_data_0 = forward_indicator(dataset, all_indices, 30)
    forward_data_1 = forward_indicator(df_indicators, indices_inds, 30)
    all_data = pd.concat([dataset, df_indicators, forward_data_0, forward_data_1], axis=1)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()]
    #all_data = all_data.dropna()
    return all_data

def forward_indicator_orig(dataset: pd.DataFrame, indices: List[str], interval: int)-> pd.DataFrame:
    dfs = [dataset]
    for i in [interval, interval + 30, interval + 60]:
        df_for = dataset[indices].shift(i)
        #df_for = df_for.fillna(method='bfill')
        #df_for = df_for.fillna(dataset[indices])
        df_for = df_for.fillna(0)
        df_for.columns = [f'{col}_FORW_{i}' for col in df_for.columns]
        dfs.append(df_for)
    df_forward = pd.concat(dfs, axis=1, join='inner')
    return df_forward

def forward_indicator(dataset: pd.DataFrame, indices: List[str], interval: int) -> pd.DataFrame:
    df_forward = dataset.copy()
    for i in [interval, interval + 30, interval + 60]:
        df_for = dataset[indices].shift(i)
        df_for = df_for.fillna(0)
        df_for.columns = [f'{col}_FORW_{i}' for col in df_for.columns]
        df_forward = pd.concat([df_forward, df_for], axis=1)
    return df_forward

