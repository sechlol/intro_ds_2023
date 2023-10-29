import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import math
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode


def convert_rankings_to_string(ranking):
    """
    Concatenate list of nodes and correlations into a single html
    string (required format for the plotly tooltip)

    Inserts html "<br>" inbetween each item in order to add a new
    line in the tooltip
    """
    s = ""
    for r in ranking:
        s += r + "<br>"
    return s


def calculate_stats(returns):
    """calculate annualised returns and volatility for all ETFs

    Returns:
        tuple: Outputs the annualised volatility and returns as a list of
            floats (for use in assigning node colours and sizes) and also
            as a lists of formatted strings to be used in the tool tips.
    """

    # log returns are additive, 252 trading days
    annualized_returns = list(np.mean(returns, axis=0) * 252 * 100)

    annualized_volatility = [
        np.std(returns[col] * 100) * (252 ** 0.5) for col in list(returns.columns)
    ]

    # create string for tooltip
    annualized_volatility_2dp = [
        "Annualized Volatility: " "%.1f" % r + "%" for r in annualized_volatility
    ]
    annualized_returns_2dp = [
        "Annualized Returns: " "%.1f" % r + "%" for r in annualized_returns
    ]

    return (
        annualized_volatility,
        annualized_returns,
        annualized_volatility_2dp,
        annualized_returns_2dp,
    )


def get_top_and_bottom_three(df):
    """
    get a list of the top 3 and bottom 3 most/least correlated assests
    for each node.

    Args:
        df (pd.DataFrame): pandas correlation matrix

    Returns:
        top_3_list (list): list of lists containing the top 3 correlations
            (name and value)
        bottom_3_list (list): list of lists containing the bottom three
            correlations (name and value)
    """

    top_3_list = []
    bottom_3_list = []

    for col in df.columns:

        # exclude self correlation #reverse order of the list returned
        top_3 = list(np.argsort(abs(df[col]))[-4:-1][::-1])
        # bottom 3 list is returned in correct order
        bottom_3 = list(np.argsort(abs(df[col]))[:3])

        # get column index
        col_index = df.columns.get_loc(col)

        # find values based on index locations
        top_3_values = [df.index[x] + ": %.2f" % df.iloc[x, col_index] for x in top_3]
        bottom_3_values = [
            df.index[x] + ": %.2f" % df.iloc[x, col_index] for x in bottom_3
        ]

        top_3_list.append(convert_rankings_to_string(top_3_values))
        bottom_3_list.append(convert_rankings_to_string(bottom_3_values))

    return top_3_list, bottom_3_list


def get_coordinates(mst):
    """Returns the positions of nodes and edges in a format
    for Plotly to draw the network
    """
    # get list of node positions
    pos = nx.fruchterman_reingold_layout(mst)

    Xnodes = [pos[n][0] for n in mst.nodes()]
    Ynodes = [pos[n][1] for n in mst.nodes()]

    Xedges = []
    Yedges = []
    for e in mst.edges():
        # x coordinates of the nodes defining the edge e
        Xedges.extend([pos[e[0]][0], pos[e[1]][0], None])
        Yedges.extend([pos[e[0]][1], pos[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def assign_colour(correlation):
    if correlation <= 0:
        return "#ffa09b"  # red
    else:
        return "#9eccb7"  # green


def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=3):
    return benchmark_thickness * abs(correlation) ** scaling_factor


def assign_node_size(degree, scaling_factor=50):
    return degree * scaling_factor


def minimum_spanning_tree(data: pd.DataFrame, date_start: str, date_end: str):
    data = data.copy()

    mask = (data.index > date_start) & (data.index <= date_end)
    data = data.loc[mask]

    init_notebook_mode(connected=True)

    log_returns_df = pd.DataFrame()
    for col in data.columns:
        log_returns_df[col] = np.log(data[col]).diff(-1)

    correlation_matrix = log_returns_df.corr()

    sns.clustermap(correlation_matrix, cmap="RdYlGn")
    # plt.show()

    # convert matrix to list of edges and rename the columns
    edges = correlation_matrix.stack().reset_index()
    edges.columns = ["asset_1", "asset_2", "correlation"]

    # remove self correlations
    edges = edges.loc[edges["asset_1"] != edges["asset_2"]].copy()

    # create undirected graph with weights corresponding to the correlation magnitude
    G0 = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

    # create subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    # save different layout functions in a list
    layouts = [nx.circular_layout, nx.random_layout, nx.spring_layout, nx.spectral_layout]

    # plot each different layout
    for layout, ax in zip(layouts, axs.ravel()):
        nx.draw(
            G0,
            with_labels=True,
            node_size=700,
            node_color="#e1575c",
            edge_color="#363847",
            pos=layout(G0),
            ax=ax,
        )
        ax.set_title(layout.__name__, fontsize=20, fontweight="bold")

    # plt.show()

    # 'winner takes all' method - set minimum correlation threshold to remove some
    # edges from the diagram
    threshold = 0.5

    # create a new graph from edge list
    Gx = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

    # list to store edges to remove
    remove = []
    # loop through edges in Gx and find correlations which are below the threshold
    for asset_1, asset_2 in Gx.edges():
        corr = Gx[asset_1][asset_2]["correlation"]
        # add to remove node list if abs(corr) < threshold
        if abs(corr) < threshold:
            remove.append((asset_1, asset_2))

    # remove edges contained in the remove list
    Gx.remove_edges_from(remove)

    # assign colours to edges depending on positive or negative correlation
    # assign edge thickness depending on magnitude of correlation
    edge_colours = []
    edge_width = []
    for key, value in nx.get_edge_attributes(Gx, "correlation").items():
        edge_colours.append(assign_colour(value))
        edge_width.append(assign_thickness(value))

    # assign node size depending on number of connections (degree)
    node_size = []
    for key, value in dict(Gx.degree).items():
        node_size.append(assign_node_size(value))

    sns.set(rc={"figure.figsize": (9, 9)})
    font_dict = {"fontsize": 18}

    nx.draw(
        Gx,
        pos=nx.circular_layout(Gx),
        with_labels=True,
        node_size=node_size,
        node_color="#e1575c",
        edge_color=edge_colours,
        width=edge_width,
    )
    plt.title("Asset price correlations", fontdict=font_dict)
    # plt.show()

    nx.draw(
        Gx,
        pos=nx.fruchterman_reingold_layout(Gx),
        with_labels=True,
        node_size=node_size,
        node_color="#e1575c",
        edge_color=edge_colours,
        width=edge_width,
    )
    plt.title("Asset price correlations - Fruchterman-Reingold layout", fontdict=font_dict)
    # plt.show()

    # create minimum spanning tree layout from Gx
    # (after small correlations have been removed)
    mst = nx.minimum_spanning_tree(Gx)

    edge_colours = []

    # assign edge colours
    for key, value in nx.get_edge_attributes(mst, "correlation").items():
        edge_colours.append(assign_colour(value))

    # draw minimum spanning tree. Set node size and width to constant
    nx.draw(
        mst,
        with_labels=True,
        pos=nx.fruchterman_reingold_layout(mst),
        node_size=200,
        node_color="#e1575c",
        edge_color=edge_colours,
        width=1.2,
    )

    # set title
    plt.title("Asset price correlations - Minimum Spanning Tree", fontdict=font_dict)
    # plt.show()

    # make list of node labels.
    node_label = list(mst.nodes())
    # calculate annualised returns, annualised volatility and round to 2dp
    annual_vol, annual_ret, annual_vol_2dp, annual_ret_2dp = calculate_stats(log_returns_df)
    # get top and bottom 3 correlations for each node
    top_3_corrs, bottom_3_corrs = get_top_and_bottom_three(correlation_matrix)

    # create tooltip string by concatenating statistics
    description = [
        f"<b>{node}</b>"
        + "<br>"
        + annual_ret_2dp[index]
        + "<br>"
        + annual_vol_2dp[index]
        + "<br><br>Strongest correlations with: "
        + "<br>"
        + top_3_corrs[index]
        + "<br>Weakest correlations with: "
          "<br>" + bottom_3_corrs[index]
        for index, node in enumerate(node_label)
    ]

    # get coordinates for nodes and edges
    Xnodes, Ynodes, Xedges, Yedges = get_coordinates(mst)

    # assign node colour depending on positive or negative annualised returns
    node_colour = [assign_colour(i) for i in annual_ret]

    # assign node size based on annualised returns size (scaled by a factor)
    node_size = [abs(x) ** 0.5 * 5 for x in annual_ret]
    node_size = [0 if math.isnan(x) else x for x in node_size]

    # Plot graph

    # edges
    tracer = go.Scatter(
        x=Xedges,
        y=Yedges,
        mode="lines",
        line=dict(color="#DCDCDC", width=1),
        hoverinfo="none",
        showlegend=False,
    )

    # nodes
    tracer_marker = go.Scatter(
        x=Xnodes,
        y=Ynodes,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=node_size, line=dict(width=1), color=node_colour),
        hoverinfo="text",
        hovertext=description,
        text=node_label,
        textfont=dict(size=7),
        showlegend=False,
    )

    axis_style = dict(
        title="",
        titlefont=dict(size=20),
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showticklabels=False,
    )

    layout = dict(
        title="Interactive minimum spanning tree: Between "+date_start+" and "+date_end,
        width=800,
        height=800,
        autosize=False,
        showlegend=False,
        xaxis=axis_style,
        yaxis=axis_style,
        hovermode="closest",
        plot_bgcolor="#fff",
    )

    fig = go.Figure()
    fig.add_trace(tracer)
    fig.add_trace(tracer_marker)
    fig.update_layout(layout)

    # fig.show()

    fig.write_html('visualizations/min_spanning_tree_('+date_start+'-'+date_end+').html')

    display(
        HTML(
            """
            <p>Node sizes are proportional to the size of
            annualised returns.<br>Node colours signify positive
            or negative returns since beginning of the timeframe.</p>
            """
        )
    )
