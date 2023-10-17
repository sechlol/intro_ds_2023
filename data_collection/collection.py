from typing import List
import pandas as pd

from . import DataSource


def _collect_from_sources(data_sources: List[DataSource]) -> List[pd.DataFrame]:
    """
        Collect data from a list of data sources and return a list of pandas DataFrames.

        This function iterates through the provided list of data sources, collects data from each source using the
        `get_data` method, and appends the resulting DataFrames to a list. Rows without data (empty DataFrames) are
        omitted from the final list.

        Parameters:
        - data_sources (List[DataSource]): A list of DataSource objects from which data will be collected.

        Returns:
        - List[pd.DataFrame]: A list of pandas DataFrames containing collected data from the data sources.

        Raises:
        - ValueError: If the `data_sources` list is empty.
    """
    if not data_sources:
        raise ValueError("data_sources is empty")

    all_data = []
    for source in data_sources:
        print(f"Collecting data from {source.name}...")
        data = source.get_data()
        if data is None:
            print("\t! Failed to collect data")
        else:
            print("\t* Collected", data.columns.to_list())
            all_data.append(data)

    return all_data


def _aggregate(data_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
        Aggregate a list of pandas DataFrames by performing inner joins on their indices.

        This function takes a list of pandas DataFrames and aggregates them by performing inner joins on their indices.
        The first DataFrame in the list is used as the starting point, and subsequent DataFrames are joined to it using
        an inner join operation, ensuring that only rows with matching index values are retained in the result DataFrame.

        Parameters:
        - data_list (List[pd.DataFrame]): A list of pandas DataFrames to be aggregated.

        Returns:
        - pd.DataFrame: A pandas DataFrame containing the result of aggregating the input DataFrames.

        Raises:
        - ValueError: If the `data_list` is empty.
    """
    if not data_list:
        raise ValueError("data_list is empty")

    return pd.concat(data_list, axis=1).dropna()


def aggregate_sources(data_sources: List[DataSource]) -> pd.DataFrame:
    all_data = _collect_from_sources(data_sources)
    joint_data = _aggregate(all_data)
    return joint_data


def aggregate_calcs(dataset: pd.DataFrame) -> pd.DataFrame:
    """""
     Not sure if necessary, but now it's here.
     Make a data frame with the computed indicators.
     Simple Moving Average, Momentum, DIY LEI, DIY LAG
    """""
    all_indices = dataset.columns.tolist()

    # Calculating indicators
    sma_df = compute_SMA(dataset, all_indices, [50])
    momentum_df = momentum(dataset, all_indices, 5)
    diy_lei_df = diy_lei(dataset)
    diy_lag_df = diy_lag(dataset)

    # Concatenating all calculated DataFrames along columns
    joint_data = pd.concat([sma_df, momentum_df, diy_lei_df, diy_lag_df], axis=1)

    joint_data_clean = joint_data.dropna()

    return joint_data_clean