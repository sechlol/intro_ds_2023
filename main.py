import pandas as pd

import data_collection as dc
import data_vis.data_exploration as de
import data_vis.result_visualization as rv
import ml.super_duper_ai as ai
import common.data_wrangling as dw
import data_vis as dv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import data_vis.data_exploration as de
import os
import ml.time_series as ts
def collect_data():
    data_sources = [
        dc.YahooDataSource(),
        dc.FredDataSource(),
        # dc.AlphaDataSource(),
        # dc.RandomDataSource(symbols=["SP500", "XLK", "XLP", "VIX", "GDP", "CSen", "10Y", "2Y"])
    ]

    dataset = dc.aggregate_sources(data_sources)
    dataset.to_csv("dataset_new.csv", float_format='%.2f')
    return dataset


def read_dataset():
    return pd.read_csv("dataset_new.csv", index_col=0, parse_dates=True)

def my_main():
    credit_crisis_0 = datetime(2007, 1,1)
    credit_crisis_1 = datetime(2009,12,31)
    pandemic_0 = datetime(2019,1,1)
    pandemic_1 = datetime(2023,10,10)
    #dataset = collect_data()
    dataset = read_dataset()
    indicators_df = dw.aggregate_calcs(dataset)
    all_data = dw.all_data(dataset)
    #spy_sta = st(['SPY'])
    #ts.sarimax_lags('SPY')
    #ts.sarimax('SPY')
    #ts.seasonal_decomposition('SPY')
    #directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    #file_name = f'all_data.txt'
    #file_path = os.path.join(directory, file_name)
    #all_data.to_csv(file_path, sep='\t', index=False)
    #spy_columns = all_data.filter(like='SPY', axis=1)
    #print(spy_columns)
    #print(all_data.shape)
    #nan_locations = all_data[all_data.isna().any(axis=1)]
    #print(nan_locations)
    #all_data = all_data.loc[:, ~all_data.columns.duplicated()]
    #all_indices = dataset.columns.tolist()
    #print(dw.forward_indicator(dataset,all_indices, 7))
    #print(dataset['SPY'])
    #dv.result_visualization.test_plot(all_data, ['T10Y2Y','PAYEMS_PCT'],'07-31-2001', '10-10-2023', False)
    #rv.split_plot(all_data, ['SPY','DCOILWTICO', 'VIXCLS'], ['UNRATE'],'07-31-2001', '10-10-2023', True)
    #rv.split_plot(all_data, ["SPY","XLE", "XLY", "XLF", "XLV","XLI","XLK","XLB","XLU","XLP"], ['LEI','LAG'], '07-31-2001', '10-10-2023', False)
    #de.monthly_correlation_slider(all_data,'DGS10', 'BAMLH0A0HYM2')
    rv.two_axes(all_data, ['PAYEMS_PCT'], ['T10Y2Y'],'06-01-2001', '10-10-2023', True)
    # "SPY","XLE", "XLY", "XLF", "XLV","XLI","XLK","XLB","XLU","XLP"


def main():
    # Get data
    # print("** DATA COLLECTION **")
    # dataset = collect_data()
    dataset = read_dataset()

    #print("\n** DATA EXPLORATION **")
    #data_vis.explore_something(dataset)
    #de.correlation_matrix(dataset)

    # Run ML models and visualize results
    #print("\n** ML MODEL TRAINING **")
    #model_result = ai.do_something(dataset)

    print("\n** RESULT VISUALIZATION **")
    rv.test_plot(dataset)


if __name__ == "__main__":
    my_main()
    #main()
