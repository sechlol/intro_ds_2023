import pandas as pd

import data_collection as dc
import data_vis
import ml.xgboost_pipeline as xgb


def collect_data():
    data_sources = [
        dc.YahooDataSource(),
        dc.AlphaDataSource(),
        dc.FredDataSource(),
        # dc.RandomDataSource(symbols=["SP500", "XLK", "XLP", "VIX", "GDP", "CSen", "10Y", "2Y"])
    ]

    dataset = dc.aggregate_sources(data_sources)
    dataset.to_csv("out/dataset.csv", float_format='%.2f')
    return dataset


def read_dataset():
    return pd.read_csv("out/dataset.csv", index_col=0, parse_dates=True)


def main():
    # Get data
    # print("** DATA COLLECTION **")
    # dataset = collect_data()
    dataset = read_dataset()

    print("\n** DATA EXPLORATION **")
    data_vis.explore_something(dataset)

    # Run ML models and visualize results
    print("\n** ML MODEL TRAINING **")
    model_result = xgb.run_pipeline(dataset)

    print("\n** RESULT VISUALIZATION **")
    data_vis.visualize_something(model_result)


if __name__ == "__main__":
    main()
