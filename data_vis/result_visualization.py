import os
from typing import List, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

from common.data_wrangling import compute_relative_strength, compute_SMA_same_shape
from data_collection import yahoo_data_source
from data_collection import fred_data_source
from main import collect_data
import plotly.graph_objects as go
from datetime import datetime
from common.data_wrangling import get_crisis_intervals


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
    date_ranges = get_crisis_intervals()
    relative_strength_data = compute_relative_strength(dataset, indices)
    sma_data = compute_SMA_same_shape(dataset, indices, window_lengths)
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'plot_{"_".join(indices)}.html'
    file_path = os.path.join(directory, file_name)
    directory_repo = r"data_vis"
    file_path_repo = os.path.join(directory_repo, file_name)
    relative_strength_data_dates = relative_strength_data.loc[start_date:end_date]
    sma_data_dates = sma_data.loc[start_date:end_date]
    fig = go.Figure()
    # Highlight business cycle date ranges
    highlight_ranges = date_ranges
    for start, end in highlight_ranges:
        fig.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="grey",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
        )
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
        fig.write_html(file_path_repo)

    fig.show()



def test_plot(dataset: pd.DataFrame, indices: List[str], start_date: datetime, end_date: datetime, save_as_html: bool = False):
    """
    General plotting
    """
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'plot_{"_".join(indices)}.html'
    file_path = os.path.join(directory, file_name)
    directory_repo = r"data_vis"
    file_path_repo = os.path.join(directory_repo, file_name)
    data = dataset.loc[start_date:end_date, indices]
    date_ranges = get_crisis_intervals()
    fig = go.Figure()
    for index in indices:
            fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index))

    # Highlight business cycle date ranges
    highlight_ranges = date_ranges
    for start, end in highlight_ranges:
        fig.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="grey",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
        )

    fig.update_layout(
        title=f'10-Year Treasury Securities and High Yield Spreads',# {", ".join(indices)}',
        xaxis_title='Date',
        yaxis_title='Percent',
    )

    if save_as_html:
        #fig.write_html(file_path)
        fig.write_html(file_path_repo)

    fig.show()
def split_plot(dataset: pd.DataFrame, indices_upper: List[str], indices_lower: List[str], start_date: datetime, end_date: datetime, save_as_html: bool = False):
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'split_plot_{"_".join(indices_upper + indices_lower)}.html'
    file_path = os.path.join(directory, file_name)
    directory_repo = r"data_vis"
    file_path_repo = os.path.join(directory_repo, file_name)
    date_ranges = get_crisis_intervals()
    data = dataset.loc[start_date:end_date,:]
    fig = go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('SPY and Market Indices','Composite Leading and Lagging Indicators'),
                        #subplot_titles=(','.join(indices_upper), f"{indices_lower[0].split('_')[0]}"),
                        row_heights=[0.6, 0.4])  # Adjust row heights as needed
    # Highlight business cycle date ranges
    highlight_ranges = date_ranges
    for start, end in highlight_ranges:
        fig.add_shape(
            dict(
                type="rect",
                xref="x1",
                yref="y1",
                x0=start,
                y0=-60,
                x1=end,
                y1=500,
                fillcolor="grey",
                opacity=0.2,
                layer="below",
                line_width=0
            )
        )
    # Highlight business cycle date ranges
    highlight_ranges = date_ranges
    for start, end in highlight_ranges:
        fig.add_shape(
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=start,
                y0=-1,
                x1=end,
                y1=10,
                fillcolor="grey",
                opacity=0.2,
                layer="below",
                line_width=0
            )
        )
    for index in indices_upper:
        fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index),
                        row=1, col=1)
    for index in indices_lower:
        fig.add_trace(go.Scatter(x=data.index, y=data[index], mode='lines', name=index),
                    row=2, col=1)

    # Update layout  ['SPY', 'WTI', 'VIX', 'UNEMP']
    #fig.update_layout(title_text=f"{', '.join(indices_upper)} with {', '.join(indices_lower)}")
    fig.update_layout(title_text='SPY and Market Indices and Composite Leading and Lagging Indicators')# 10-Year Constant Maturity Treasury Securities Yield and High Yield Spread')#WTI (DCOILWTICO), VIX (VIXCLS) and Unemployment Rate)
    if save_as_html:
        #fig.write_html(file_path)
        fig.write_html(file_path_repo)
    # Show figure
    fig.show()



def two_axes(dataset: pd.DataFrame, index_1: Union[str, List[str]], index_2: Union[str, List[str]],
             start_date: datetime, end_date: datetime, save_as_html: bool = False):
    directory = r"C:\Users\elvad\Documents\Intro_to_DS"
    file_name = f'2axes_plot_{"_".join(index_1 + index_2)}.html'
    file_path = os.path.join(directory, file_name)
    directory_repo = r"data_vis"
    file_path_repo = os.path.join(directory_repo, file_name)

    # Assuming read_dates() returns a list of tuples with start and end dates
    date_ranges = get_crisis_intervals()

    data = dataset.loc[start_date:end_date, :]

    # Create subplots and set the second y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Highlight business cycle date ranges
    for start, end in date_ranges:
        fig.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="grey",
                opacity=0.2,
                layer="below",
                line_width=0
            )
        )

    # Add traces
    if isinstance(index_1, list):
        for idx in index_1:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[idx], name='All_Emps (Left)'),
                secondary_y=False,
            )
    else:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[index_1], name=f'{index_1} (Left)'),
            secondary_y=False,
        )

    if isinstance(index_2, list):
        for idx in index_2:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[idx], name=f'{idx} (Right)'),
                secondary_y=True,
            )
    else:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[index_2], name=f'{index_2} (Right)'),
            secondary_y=True,
        )

    # Set axis labels and title
    fig.update_layout(
        title_text="10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity and All Employees,Total Non-Farm",
        xaxis_title="Time",
        yaxis_title="Percent Change from Year Ago",
        yaxis2_title="Percent",
    )

    if save_as_html:
        #fig.write_html(file_path)
        fig.write_html(file_path_repo)

    # Show the plot
    fig.show()
