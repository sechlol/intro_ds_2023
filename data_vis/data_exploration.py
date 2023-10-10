import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def explore_something(dataset: pd.DataFrame):
    print("Much exploration! Very data! Oh Yeah!")

def correlation_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = dataset.corr()

    fig, ax = plt.subplots(figsize=(15, 12))
    cmap = sb.diverging_palette(20, 220, as_cmap=True)
    dataplot = sb.heatmap(corr_matrix,
                          cmap=cmap,
                          annot=True,
                          vmin=-1,
                          vmax=1,
                          fmt='.2f')
    plt.title('Correlation matrix')
    plt.savefig('visualizations/correlation_matrix.png', bbox_inches='tight')

    return corr_matrix
