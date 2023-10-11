from typing import Optional

import pandas as pd
from common import data_wrangling as dw
from . import xgboost_pipeline as xgb
from . import mlp_pipeline as mlp

_INDICES = ["SPY", "XLE", "XLY", "XLF", "XLV", "XLI", "XLK", "XLB", "XLU", "XLP"]


def do_something(dataset: pd.DataFrame) -> pd.DataFrame:
    data_enriched = _preprocess_data(dataset)
    y_target = _make_target(data_enriched, periods_difference=20)

    _do_mlp(data_enriched, y_target)
    print()
    _do_xgb(data_enriched, y_target)

    # Return empty result for now.
    return pd.DataFrame()


def _do_mlp(data_enriched, y_target):
    print("*** Multilayer perceptron pipeline *** ")
    x_pre_2022 = data_enriched[:"2021"]
    y_pre_2022 = y_target[:"2021"].to_numpy()
    x_post_2022 = data_enriched["2022":].to_numpy()
    y_post_2022 = y_target["2022":].to_numpy()

    train_result = mlp.run_pipeline(x_pre_2022, y_pre_2022, cross_validate=False)
    prediction = mlp.make_predictions(x_post_2022, y_post_2022, train_result.model)
    print(prediction)


def _do_xgb(data_enriched, y_target):
    print("*** XGBoost pipeline *** ")
    x_pre_2022 = data_enriched[:"2021"]
    y_pre_2022 = y_target[:"2021"].to_numpy()
    x_post_2022 = data_enriched["2022":].to_numpy()
    y_post_2022 = y_target["2022":].to_numpy()

    train_result = xgb.run_pipeline(x_pre_2022, y_pre_2022, cross_validate=False)
    prediction = xgb.make_predictions(x_post_2022, y_post_2022, train_result.booster)
    print(prediction)


def _preprocess_data(dataset: pd.DataFrame, resample_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesses a given dataset by computing Simple Moving Averages (SMAs), relative strengths,
    and optionally resampling the data to a specified frequency.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing historical data for stock indices.
        resample_freq (Optional[str]): The frequency at which to resample the data.
            Possible values are 'W' (for weekly) and 'M' (for monthly). If None, no resampling is performed.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with SMAs, relative strengths, and optional resampling.
    """
    smas = dw.compute_SMA(dataset, indices=_INDICES, window_lengths=[20, 50, 200])
    rel_strengths = dw.compute_relative_strength(dataset, _INDICES)
    combined = pd.concat([dataset, rel_strengths, smas], axis=1).dropna()
    if resample_freq:
        return combined.resample(resample_freq).last()
    else:
        return combined


def _make_target(dataset: pd.DataFrame, periods_difference: int) -> pd.DataFrame:
    """
    Creates a binary target variable based on the price movement of the target index.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing historical data for stock indices.
        periods_difference (int): The number of periods to look ahead for computing the target variable.
            A positive value indicates looking into the future, while a negative value looks into the past.

    Returns:
        pd.DataFrame: A DataFrame containing the binary target variable.
            - 1 indicates that the "SPY" index increased over the specified number of periods.
            - 0 indicates that the "SPY" index either remained unchanged or decreased over the specified periods.
    """
    y = dataset["SPY"] < dataset["SPY"].shift(periods=-periods_difference)
    return y.to_frame()
