import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.data_wrangling import compute_relative_strength, compute_SMA
from data_collection import yahoo_data_source
from data_collection import fred_data_source
from main import collect_data
import plotly.graph_objects as go

def visualize_something(dataset: pd.DataFrame):
    print("Results are WOW!")
    print(dataset)


def RS_plot(dataset: pd.DataFrame, indices: List[str], save_as_html: bool = False):
    """
    Plotting the relative strength
    """
    file_name = f'RSI_{"_".join(indices)}.html'
    window_lengths = [30, 90, 180]

    relative_strength_data = compute_relative_strength(dataset, indices)
    sma_data = compute_SMA(dataset, indices, window_lengths)

    # Create a plotly figure for saving
    fig = go.Figure()

    for col in relative_strength_data.columns:
        fig.add_trace(go.Scatter(x=relative_strength_data.index,
                                 y=relative_strength_data[col],
                                 mode='lines',
                                 name=col))

    fig.update_layout(title=f'Relative Strength of {indices[0]} index and {indices[1]}',
                      xaxis_title='Date',
                      yaxis_title='Normalized Relative Strength')

    if save_as_html:
        fig.write_html(file_name)

    plt.show()




def test_plot(dataset: pd.DataFrame, indices: List[str], start_date: datetime, end_date: datetime, save_as_html: bool = False):
    """
    General plotting
    """
    file_name = f'RSI_{"_".join(indices)}.html'
    data = dataset.loc[start_date:end_date, indices]

    # Create a plotly figure for saving
    fig = go.Figure()
    plt.figure(figsize=(12, 6))
    for index in indices:
        plt.plot(data)

    plt.title(f'Time Series of {", ".join(indices)}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(data.columns)

    if save_as_html:
        fig.write_html(file_name)

    plt.show()