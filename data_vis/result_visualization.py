import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

from common.data_wrangling import compute_relative_strength, compute_SMA
from data_collection import yahoo_data_source
from data_collection import fred_data_source
from main import collect_data
import plotly.graph_objects as go
from datetime import datetime



def visualize_something(dataset: pd.DataFrame):
    print("Results are WOW!")
    print(dataset)


def RS_plot(dataset: pd.DataFrame, indices: List[str], start_date: datetime, end_date: datetime,
            save_as_html: bool = False):
    """
    Plotting the relative strength
    """
    file_name = f'RSI_{"_".join(indices)}.html'
    window_lengths = [30, 90, 180]

    relative_strength_data = compute_relative_strength(dataset, indices)
    sma_data = compute_SMA(dataset, indices, window_lengths)
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'plot_{"_".join(indices)}.html'
    file_path = os.path.join(directory, file_name)
    relative_strength_data_dates = relative_strength_data.loc[start_date:end_date]
    sma_data_dates = sma_data.loc[start_date:end_date]
    fig = go.Figure()

    for col in relative_strength_data.columns:
        fig.add_trace(go.Scatter(x=relative_strength_data_dates.index,
                                 y=relative_strength_data_dates[col],
                                 mode='lines',
                                 name=col))
    fig.update_layout(
        title=f'Relative Strength of {indices[0]} / index and {indices[1]}',
        xaxis_title='Date',
        yaxis_title='Normalized Relative Strength',
    )

    if save_as_html:
        fig.write_html(file_path)

    fig.show()



def test_plot(dataset: pd.DataFrame, indices: List[str], start_date: datetime, end_date: datetime, save_as_html: bool = False):
    """
    General plotting
    """
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'plot_{"_".join(indices)}.html'
    file_path = os.path.join(directory, file_name)
    data = dataset.loc[start_date:end_date, indices]
    fig = go.Figure()
    for index in indices:
            fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index))

    fig.update_layout(
        title=f'Time Series of {", ".join(indices)}',
        xaxis_title='Date',
        yaxis_title='Returns',
    )

    if save_as_html:
        fig.write_html(file_path)

    fig.show()

def split_plot(dataset: pd.DataFrame, indices_upper: List[str], indices_lower: List[str], start_date: datetime, end_date: datetime, save_as_html: bool = False):
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'split_plot_{"_".join(indices_upper + indices_lower)}.html'
    file_path = os.path.join(directory, file_name)
    data = dataset.loc[start_date:end_date,:]
    fig = go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(','.join(indices_upper) , f"Momentum of {indices_lower[0].split('_')[0]}"),
                        row_heights=[0.6, 0.4])  # Adjust row heights as needed
    for index in indices_upper:
        fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index),
                    row=1, col=1)
    for index in indices_lower:
        fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index),
                    row=2, col=1)

    # Update layout
    fig.update_layout(title_text=f"{', '.join(indices_upper)} with {', '.join(indices_lower)}")
    if save_as_html:
        fig.write_html(file_path)
    # Show figure
    fig.show()
