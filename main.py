import pandas as pd

import data_collection as dc
import data_vis

import data_vis.data_exploration as de
import ml.super_duper_ai as ai


def collect_data():
    data_sources = [
        dc.YahooDataSource(),
        dc.FredDataSource(),
        # dc.AlphaDataSource(),
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
    de.correlation_matrix(dataset)

    # Run ML models and visualize results
    print("\n** ML MODEL TRAINING **")
    model_result = ai.do_something(dataset)

    print("\n** RESULT VISUALIZATION **")
    data_vis.visualize_something(model_result.history)


if __name__ == "__main__":
    main()
