import pathlib

import pandas as pd

import data_collection as dc
import data_vis.data_exploration as de
import ml.super_duper_ai as ai

_DATASET_PATH = pathlib.Path("out/dataset.csv")


def collect_data():
    data_sources = [
        dc.YahooDataSource(),
        dc.FredDataSource(),
        dc.AlphaDataSource(),
        # dc.CsvDataSource(path=_DATASET_PATH),
    ]

    dataset = dc.aggregate_sources(data_sources)

    # Save dataset to file. Make sure that directory exists
    _DATASET_PATH.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(_DATASET_PATH, float_format='%.2f')

    return dataset


def read_dataset():
    return pd.read_csv(_DATASET_PATH, index_col=0, parse_dates=True)


def main():
    # Get data
    print("** DATA COLLECTION **")
    dataset = read_dataset() if _DATASET_PATH.exists() else collect_data()
    
    print("\n** DATA EXPLORATION **")
    de.explore_data(dataset)

    # Run ML models and visualize results
    print("\n** ML MODEL TRAINING **")
    ai.run_pipelines(dataset)


if __name__ == "__main__":
    main()
