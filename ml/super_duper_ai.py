from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import data_wrangling as dw
from . import xgboost_pipeline as xgb
from . import mlp_pipeline as mlp
from . import lstm_pipeline as lstm

_INDICES = ["SPY", "XLE", "XLY", "XLF", "XLV", "XLI", "XLK", "XLB", "XLU", "XLP"]


def do_something(dataset: pd.DataFrame) -> pd.DataFrame:
    predict_feature = "XLP"
    predict_ahead = 10
    data_enriched = _preprocess_data(dataset, predict_feature, predict_ahead, scale=True)
    y_target = _make_target(data_enriched, predict_feature, periods_difference=predict_ahead)

    # _do_xgb(data_enriched, y_target)
    # print()
    # _do_mlp(data_enriched, y_target)
    # print()
    _do_lstm(data_enriched, y_target)

    # Return empty result for now.
    return pd.DataFrame()


def _do_lstm(data_enriched, y_target):
    print("*** LSTM pipeline *** ")
    x_pre_2022 = data_enriched[:"2021"]
    y_pre_2022 = y_target[:"2021"].to_numpy()
    x_post_2022 = data_enriched["2022":].to_numpy()
    y_post_2022 = y_target["2022":].to_numpy()

    model, history = lstm.run_pipeline(x_pre_2022, y_pre_2022)
    train_data = lstm.make_predictions(x_pre_2022.to_numpy(), y_pre_2022, model)
    test_data = lstm.make_predictions(x_post_2022, y_post_2022, model)
    lstm.save_result(train_data, test_data, history, split=(x_pre_2022, x_post_2022, y_pre_2022, y_post_2022))


def _do_mlp(data_enriched, y_target):
    print("*** Multilayer perceptron pipeline *** ")
    x_pre_2022 = data_enriched[:"2021"]
    y_pre_2022 = y_target[:"2021"].to_numpy()
    x_post_2022 = data_enriched["2022":].to_numpy()
    y_post_2022 = y_target["2022":].to_numpy()

    train_result = mlp.run_pipeline(x_pre_2022, y_pre_2022, cross_validate=False)
    mlp.make_predictions(x_post_2022, y_post_2022, train_result.model)


def _do_xgb(data_enriched, y_target):
    print("*** XGBoost pipeline *** ")
    x_pre_2022 = data_enriched[:"2021"]
    y_pre_2022 = y_target[:"2021"].to_numpy()
    x_post_2022 = data_enriched["2022":].to_numpy()
    y_post_2022 = y_target["2022":].to_numpy()

    train_result = xgb.run_pipeline(x_pre_2022, y_pre_2022, cross_validate=False)
    xgb.make_predictions(x_post_2022, y_post_2022, train_result.booster)


def _preprocess_data(
        dataset: pd.DataFrame,
        feature: str,
        predict_ahead: int,
        resample_freq: Optional[str] = None,
        scale: bool = False) -> pd.DataFrame:
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
    rel_returns = dw.compute_returns(dataset, _INDICES, period_differences=[predict_ahead, 20, 50, 200])

    # Keep only one feature from the original prices. Discard all the others
    combined = (pd
                .concat([dataset, smas, rel_returns, rel_strengths], axis=1)
                .dropna()
                .drop(columns=[i for i in _INDICES if i is not feature]))

    if resample_freq:
        combined = combined.resample(resample_freq).last()
    if scale:
        transformed = StandardScaler().fit_transform(combined)
        combined = pd.DataFrame(transformed, columns=combined.columns.values, index=combined.index)

    return combined


def _make_target(dataset: pd.DataFrame, feature: str, periods_difference: int) -> pd.DataFrame:
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
    y = dataset[feature] < dataset[feature].shift(periods=-periods_difference)
    return y.to_frame()
