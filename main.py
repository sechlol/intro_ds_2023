import pandas as pd

import data_collection as dc
import data_vis

import data_vis.data_exploration as de
import data_vis.minimum_spanning_tree as mst
import ml.super_duper_ai as ai

_DATASET_PATH = "out/dataset.csv"


def collect_data():
    data_sources = [
        dc.CsvDataSource(path=_DATASET_PATH),
        # dc.YahooDataSource(),
        # dc.FredDataSource(),
        # dc.AlphaDataSource(),
    ]

    dataset = dc.aggregate_sources(data_sources)
    dataset.to_csv(_DATASET_PATH, float_format='%.2f')
    return dataset


def read_dataset():
    return pd.read_csv(_DATASET_PATH, index_col=0, parse_dates=True)


def main():
    # Get data
    # print("** DATA COLLECTION **")
    # dataset = collect_data()
    dataset = read_dataset()

    print("\n** DATA EXPLORATION **")
    de.correlation_matrix(dataset)
    de.relative_returns(dataset)
    de.correlation_slider(dataset, 'XLE', 'XLY')
    mst.minimum_spanning_tree(dataset, '2015-12-1', '2016-12-1')

    # Run ML models and visualize results
    print("\n** ML MODEL TRAINING **")
    #model_result = ai.do_something(dataset)

    # print("\n** RESULT VISUALIZATION **")
    # data_vis.visualize_something(model_result.history)


if __name__ == "__main__":
    main()
