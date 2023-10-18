from pathlib import Path

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

from common import data_wrangling as dw

_OUT_PATH = Path("visualizations")


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = data.corr()

    cmap = sb.diverging_palette(5, 260, as_cmap=True)
    plt.figure(figsize=(15, 12))
    sb.heatmap(corr_matrix,
               cmap=cmap,
               annot=True,
               vmin=-1,
               vmax=1,
               fmt='.2f')

    plt.title('Correlation matrix')
    plt.savefig(_OUT_PATH / 'correlation_matrix.png', bbox_inches='tight')

    return corr_matrix


def relative_returns(dataset: pd.DataFrame, period_days: int = 30):
    indices = ["SPY", "XLE", "XLY", "XLF", "XLV", "XLI", "XLK", "XLB", "XLU", "XLP"]
    returns = dw.compute_returns(dataset, indices, [period_days])

    fig, axes = plt.subplots(5, len(indices)//5, figsize=(20, 15))
    fig.suptitle(f"Relative return at {period_days} days for indices, in %")

    for index, ax in zip(returns, axes.flatten()):
        title = index.split("_")[0]
        ax.set(title=title)
        sb.lineplot(returns[index]*100, ax=ax)

    fig.tight_layout()
    fig.savefig(_OUT_PATH / 'returns.png', bbox_inches='tight')


def monthly_correlation_slider(dataset: pd.DataFrame, ind1: str, ind2: str):

    # Drop elements that have no daily data
    dataset = dataset.copy()
    dataset.drop(['GDPC1', 'UNRATE', 'INDPRO', 'PAYEMS', 'UMCSENT', 'FEDFUNDS', 'CPIAUCSL', 'PPIACO',
                            'NEWORDER', 'AWHMAN', 'NEWORDER', 'ACOGNO', 'CES4348400001'],
                 axis=1,
                 inplace=True)

    # Calculate the correlation
    df_corr = dataset[[ind1, ind2]].groupby([(dataset[[ind1, ind2]].index.year), (dataset[[ind1, ind2]].index.month)]).corr()
    corr = df_corr[ind1].to_numpy()[1::2]

    # Stupid stuff to get datetime.date
    df_corr = df_corr.reset_index(level=[2])
    times = df_corr.index.to_flat_index().to_numpy()[0::2]
    year = [t[0] for t in times]
    month = [t[1] for t in times]
    day = np.ones(len(month), dtype=np.int8)
    dates = [str(year[i])+'-'+str(month[i])+'-'+str(day[i]) for i in range(len(month))]
    dates = [datetime.strptime(dates[i], '%Y-%m-%d').date() for i in range(len(month))]

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    fig.add_trace(
        go.Scatter(x=list(dates), y=list(corr)))

    # Set title
    fig.update_layout(
        title_text='Correlation between '+str(ind1)+' and '+str(ind2)
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # fig.show()

    return fig.write_html('visualizations/correlation_'+str(ind1)+'_and_'+str(ind2)+'.html')


def yearly_correlation_slider(dataset: pd.DataFrame, ind1: str, ind2: str):

    dataset = dataset.copy()

    # Calculate the correlation
    df_corr = dataset[[ind1, ind2]].groupby([(dataset[[ind1, ind2]].index.year)]).corr()
    corr = df_corr[ind1].to_numpy()[1::2]

    # Stupid stuff to get datetime.date
    df_corr = df_corr.reset_index(level=[1])
    year = df_corr.index.to_flat_index().to_numpy()[0::2]
    month = np.ones(len(year), dtype=np.int8)
    day = np.ones(len(year), dtype=np.int8)
    dates = [str(year[i]) + '-' + str(month[i]) + '-' + str(day[i]) for i in range(len(month))]
    dates = [datetime.strptime(dates[i], '%Y-%m-%d').date() for i in range(len(month))]

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    fig.add_trace(
        go.Scatter(x=list(dates), y=list(corr)))

    # Set title
    fig.update_layout(
        title_text='Correlation between ' + str(ind1) + ' and ' + str(ind2)
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=5,
                         label="5y",
                         step="year",
                         stepmode="backward"),
                    dict(count=10,
                         label="10y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # fig.show()

    return fig.write_html('visualizations/correlation_' + str(ind1) + '_and_' + str(ind2) + '.html')


def correlation_slider(dataset: pd.DataFrame, ind1: str, ind2: str):
    """

    Args: Create a slider to show the correlation between two indices.
            ind1: If you select one of the following indices:
        ['GDPC1', 'UNRATE', 'INDPRO', 'PAYEMS', 'UMCSENT', 'FEDFUNDS', 'CPIAUCSL', 'PPIACO',
        'NEWORDER', 'AWHMAN', 'NEWORDER', 'ACOGNO', 'CES4348400001'], you will get yearly slider.

            ind2: If you select one of the following indices:
        ['GDPC1', 'UNRATE', 'INDPRO', 'PAYEMS', 'UMCSENT', 'FEDFUNDS', 'CPIAUCSL', 'PPIACO',
        'NEWORDER', 'AWHMAN', 'NEWORDER', 'ACOGNO', 'CES4348400001'], you will get yearly slider.

    Returns: A slider plot to show the correlation between two indices.

    """
    monthly_data_sources = ['GDPC1', 'UNRATE', 'INDPRO', 'PAYEMS', 'UMCSENT', 'FEDFUNDS', 'CPIAUCSL', 'PPIACO',
                            'NEWORDER', 'AWHMAN', 'NEWORDER', 'ACOGNO', 'CES4348400001']
    if (ind1 in monthly_data_sources) or (ind2 in monthly_data_sources):
        yearly_correlation_slider(dataset, ind1, ind2)
    else:
        monthly_correlation_slider(dataset, ind1, ind2)
