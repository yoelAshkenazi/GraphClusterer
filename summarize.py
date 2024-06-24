"""
Yoel Ashkenazi
this file is responsible for summarizing the result graphs.
"""
import os
import pickle as pkl
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return:
    """
    file_path = f"data/processed_graphs/{name}.pkl"
    with open(file_path, 'rb') as f:
        graph = pkl.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(f"Successfully removed {s-len(nodes)} nan {'vertex' if s-len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)
    return graph


def filter_by_colors(graph: nx.Graph) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to the community colors.
    :param graph: the graph.
    :return:
    """
    # first we filter articles by vertex type.
    articles = [node for node in graph.nodes() if graph.nodes.data()[node]['type'] == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    nodes = graph.nodes(data=True)
    colors = set([node[1]['color'] for node in nodes])
    subgraphs = []
    for color in colors:  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]
        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph]):
    """
    This method summarizes each of the subgraphs' abstract texts using PRIMER, prints the results and save them
    to a .txt file.
    :param subgraphs: List of subgraphs.
    :return:
    """