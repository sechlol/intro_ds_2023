import data_collection as dc
import data_vis
import data_vis.data_exploration as de
import ml.super_duper_AI as ai


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


def main():
    # Get data and explore it
    print("** DATA COLLECTION **")
    dataset = collect_data()

    print("\n** DATA EXPLORATION **")
    data_vis.explore_something(dataset)
    de.correlation_matrix(dataset)

    # Run ML models and visualize results
    print("\n** ML MODEL TRAINING **")
    model_result = ai.do_something(dataset)

    print("\n** RESULT VISUALIZATION **")
    data_vis.visualize_something(model_result)


if __name__ == "__main__":
    main()
