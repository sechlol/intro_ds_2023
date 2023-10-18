from pathlib import Path

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from common import data_wrangling as dw

_OUT_PATH = Path("out/visualizations")


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
