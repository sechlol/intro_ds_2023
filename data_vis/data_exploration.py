from pathlib import Path

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

_OUT_PATH = Path("out/visualizations")


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = data.corr()

    fig, ax = plt.subplots(figsize=(15, 12))
    cmap = sb.diverging_palette(5, 260, as_cmap=True)
    dataplot = sb.heatmap(corr_matrix,
                          cmap=cmap,
                          annot=True,
                          vmin=-1,
                          vmax=1,
                          fmt='.2f')

    _OUT_PATH.mkdir(parents=True, exist_ok=True)
    plt.title('Correlation matrix')
    plt.savefig(_OUT_PATH / 'correlation_matrix.png', bbox_inches='tight')

    return corr_matrix
