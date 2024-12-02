"""
Yoel Ashkenazi
Clustering the graphs using the original edges and similarity edges based on the distances.
"""
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pk
import random
import os

wikipedia = False


def update_wikipedia():
    global wikipedia
    wikipedia = True


# load a .pkl file.
def load_pkl(file_path):
    """
    Load a .pkl file.
    :param file_path: the path to the file.
    :return:
    """
    with open(file_path, 'rb') as f:
        data = pk.load(f)
    return data


def get_file_path(name):
    """
    Get the file path for the embeddings of the given name.
    :param name: file name.
    :return:
    """
    return 'data/distances/' + name + '_papers_embeddings.pkl'


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name. Graphs are saved in the 'data/clustered_graphs' directory.
    :param name: the name of the graph.
    :return: the graph.
    :return:
    """
    graph_path = f'data/clustered_graphs/{name}.gpickle'  # the path to the graph.

    # load the graph.
    with open(graph_path, 'rb') as f:
        graph = pk.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    return graph


def make_graph(vertices, edges, distance_matrix=None, **kwargs):
    """
    Create a graph with the given parameters.
    :param vertices: the vertices of the graph.
    :param edges: the edges of the graph.
    :param distance_matrix: the distances between the vertices.
    :param kwargs: additional arguments for the graph.
    :return:

    ------------
    Example:
    ------------
    name = '3D printing'

    graph_kwargs = {'A': 1.0, 'size': 200, 'color': '#1f78b4', 'distance_threshold': 0.5, distance_matrix: dists}

    G = functions.make_graph('3D printing', **graph_kwargs)

    this will create a graph for the '3D printing' embeddings, with the given vertex sizes and colors.
    """
    # Unpack the parameters.
    vertex_size = kwargs['size'] if 'size' in kwargs else 200
    vertex_color = kwargs['color'] if 'color' in kwargs else '#1f78b4'
    K = kwargs['K'] if 'K' in kwargs else 1.0  # Knn parameter.

    # Add a default shape mapping.
    type_to_shape_map = {'paper': 's', 'author': '*', 'keyword': 'd', 'institution': 'p', 'country': 'o'}

    # create a graph.
    G = nx.Graph()

    # add the vertices to the graph.
    for i, vertex in vertices.iterrows():

        G.add_node(vertex['id'], size=vertex_size, shape=type_to_shape_map['paper'], type='paper',
                   color=vertex_color)

    # add the edges to the graph. (blue edges - original)
    # divide into two cases: 2 or 3 columns. if 2 columns, means all the edge ends are already in the vertices list.
    # if 3 columns, means we need to add the edge ends to the vertices list.
    if len(edges.columns) == 2:
        for i in range(len(edges)):
            edge = edges.iloc[i, :]
            G.add_edge(edge[0], edge[1], weight=1, color='blue')

    else:
        sources = edges.iloc[:, 1]
        targets = edges.iloc[:, 2]
        types = edges.iloc[:, 3]

        # Go over the edges and add them to the graph.
        for i in range(len(sources)):
            if targets[i] is None:  # if the target is None, skip the edge.
                continue

            if targets[i] not in G.nodes():  # add the target to the graph.
                G.add_node(targets[i], size=vertex_size, shape=type_to_shape_map[types[i]], type=types[i],
                           color=vertex_color)
            G.add_edge(sources[i], targets[i], weight=1, color='blue')

    # add the similarity edges to the graph. (red edges if provided)
    if distance_matrix is not None:
        np_dists = np.array(distance_matrix)
        np_dists = np.argsort(np_dists, axis=1)[:, 1:K + 1]  # get the K nearest neighbors.
        for i, vertex in vertices.iterrows():
            for j in np_dists[i]:  # add the edges to the graph.
                G.add_edge(vertex['id'], vertices.iloc[j]['id'], weight=1, color='red')

    return G


def draw_graph(G, name, **kwargs):
    """
    Draw a graph with the given colors and shapes.
    :param G: the graph to draw.
    :param name: the name of the graph.
    :return:

    ------------
    Example:
    ------------
    name = '3D printing'

    draw_kwargs = {'shown_percentage': 0.65, 'figsize': 25, 'save': True, 'method': 'louvain'}
    functions.draw_graph(G, **draw_kwargs)
    this will draw the graph for the '3D printing' embeddings, with the given parameters.
    """

    rate = kwargs['shown_percentage'] if 'shown_percentage' in kwargs else 1
    figsize = kwargs['figsize'] if 'figsize' in kwargs else 8
    save = kwargs['save'] if 'save' in kwargs else False
    method = kwargs['method'] if 'method' in kwargs else 'louvain'

    plt.figure(figsize=(figsize, figsize))
    pos = nx.spring_layout(G)  # positions for all nodes.

    # randomly select a rate of vertices to draw.
    vertices = random.sample(list(G.nodes), int(rate * len(G.nodes()))) if rate < 1 else list(G.nodes)
    G = G.subgraph(vertices) if rate < 1 else G

    blue_weights = [G[u][v]['weight'] for u, v in G.edges() if G[u][v]['color'] == 'blue']  # weights for blue edges.
    red_weights = [G[u][v]['weight'] for u, v in G.edges() if G[u][v]['color'] == 'red']  # weights for red edges.
    # draw vertices with given shapes and sizes.
    shapes = nx.get_node_attributes(G, 'shape').values()
    # draw the vertices according to shapes.
    for shape in shapes:
        # get a list of nodes with the same shape.
        vertices_ = [v for v in G.nodes if
                     shape == G.nodes()[v]['shape']]
        # get a list of the sizes of said nodes.
        vertex_sizes = [G.nodes()[v]['size'] for v in vertices_]
        # get a list of the colors of said nodes.
        vertex_colors = [G.nodes()[node]['color'] for node in vertices_]
        nx.draw_networkx_nodes(G, pos, nodelist=vertices_, node_size=vertex_sizes,
                               node_shape=shape, node_color=vertex_colors, alpha=0.8,
                               linewidths=1.5, edgecolors='black')  # always draw a black border.

    # draw the blue edges.
    if len(blue_weights) > 0:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'blue'],
                               edge_color='blue', width=blue_weights, alpha=0.75)
    # draw the red edges. (if they exist)
    if len(red_weights) > 0:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'red'],
                               edge_color='red', width=red_weights, alpha=0.75)

    if save:
        filename = f'Figures/{int(100 * rate)}_percents_shown/{method}_method/{name}'
        try:
            plt.savefig(f'{filename}.png')
        except FileNotFoundError:  # create the directory if it doesn't exist.
            import os
            os.makedirs(f'Figures/{int(100 * rate)}_percents_shown/{method}_method/')
            plt.savefig(f'{filename}.png')
    plt.show()


def cluster_graph(G, name, **kwargs):
    """
    divide the vertices of the graph into clusters.
    set a random color to each cluster's nodes.
    :param G: the graph to cluster.
    :param name: the name of the graph.
    :param kwargs: additional arguments for the clustering method.
    :return:
    """

    global wikipedia
    # Unpack the parameters.
    method = kwargs['method'] if 'method' in kwargs else 'louvain'
    save = kwargs['save'] if 'save' in kwargs else False
    res = kwargs['resolution'] if 'resolution' in kwargs else 1.0

    # cluster the graph.
    if method == 'louvain':
        partition = nx.algorithms.community.louvain_communities(G, resolution=res)
    else:
        raise ValueError(f"Clustering method '{method}' is not supported.")

    # set a random color to each cluster.
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(partition))]
    for i, cluster in enumerate(partition):
        for node in cluster:
            G.nodes[node]['color'] = colors[i]

    if save:
        filename = f'data/clustered_graphs/{name}.gpickle'  # save the graph to a file.

        try:  # save the graph.
            with open(filename, 'wb') as f:
                pk.dump(G, f, protocol=pk.HIGHEST_PROTOCOL)
        except OSError:  # create the directory if it doesn't exist.
            os.makedirs('data/clustered_graphs/')
            with open(filename, 'wb') as f:
                pk.dump(G, f, protocol=pk.HIGHEST_PROTOCOL)

        print(f"Graph for '{name}' saved successfully to '{filename}'.")

    return partition, G


def analyze_clusters(name):
    """
    Analyze the clusters of the graph.
    :param name: the name of the dataset.
    :return: a list of cluster sizes.
    """

    # load the graph.
    G = load_graph(name)

    # first we filter articles by vertex type.
    articles = [node for node in G.nodes if G.nodes()[node]['type'] == 'paper']
    articles_graph = G.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    colors = set([graph.nodes()[node]['color'] for node in graph.nodes])
    sizes = []
    for color in colors:  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]
        sizes.append(len(nodes))

    return sizes


def evaluate_clusters(name, distance_matrix=None):
    """
    Evaluate the clusters of the graph.
    :param name:  the name of the dataset.
    :param distance_matrix:  the distance matrix.
    :return:  The largest cluster percentage, and the average cluster distance index.
    """
    sizes = analyze_clusters(name)
    largest_cluster_percentage = max(sizes) / sum(sizes)  # get the largest cluster percentage.

    if distance_matrix is None:  # if the distance matrix is not provided, return only the largest cluster percentage.
        return largest_cluster_percentage

    # get the average cluster distance index.
    # first we load the graph.
    G = load_graph(name)
    articles = [node for node in G.nodes if G.nodes.data()[node]['type'] == 'paper']
    articles_graph = G.subgraph(articles)
    vertex_indices = {node: i for i, node in enumerate(articles_graph.nodes)}
    colors = set([node[1]['color'] for node in articles_graph.nodes(data=True)])  # get the colors.
    total_d_shared_clusters = 0  # Total distance between vertices that share clusters.
    for color in colors:
        cluster = [vertex_indices[v] for v in articles_graph.nodes if articles_graph.nodes[v]['color'] == color]
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                total_d_shared_clusters += distance_matrix[cluster[i], cluster[j]]

    # Get the total average distance (average distance between any two nodes in the graph).
    triu_dists = np.triu(distance_matrix, 0)
    avg_index = total_d_shared_clusters / np.sum(triu_dists)  # get the average cluster distance index.

    return avg_index, largest_cluster_percentage


def check_weight_prop(G, start, end, step, name, res, repeat):
    """
    takes a proportion graph, copies it and changes the weights for each edge type:
    1. original edges. set to 1
    2. similarity edges. set to 10**-x for x in range(start, end, step)
    then re-clusters the graph and returns the average index and largest cluster percentage for each weight.
    :param G:  the graph.
    :param start:  the start of the range.
    :param end:  the end of the range.
    :param step:  the step of the range.
    :param name:  the name of the dataset.
    :param res:  the resolution coefficient.
    :param repeat:  the number of times to repeat the process.
    :return:
    """

    # initialize the variables.
    final_avg_indexes = []
    final_largest_cluster_percentages = []
    final_std_indexes = []
    final_std_percentages = []

    # copy the graph.
    G_copy = copy.deepcopy(G)
    x_lst = np.linspace(start, end, step)
    x_lst = [10 ** -x for x in x_lst]
    # change the weights for the edges.

    for i in x_lst:
        for u, v in G_copy.edges():
            if G_copy[u][v]['color'] == 'blue':
                G_copy[u][v]['weight'] = 1
            else:
                G_copy[u][v]['weight'] = 10 ** -i
        avg_indexes = []
        largest_cluster_percentages = []
        for j in range(repeat):
            # re-cluster the graph.
            cluster_graph(G_copy, name, method='louvain', resolution=res, save=False)
            # evaluate the clusters.
            avg_index, largest_cluster_percentage = evaluate_clusters(name)
            avg_indexes.append(avg_index)
            largest_cluster_percentages.append(largest_cluster_percentage)

        # get the average and standard deviation for the indexes and percentages,
        # meaning for each weight we get the average and standard deviation of the indexes and percentages.
        final_avg_indexes.append(round(np.mean(avg_indexes), 3))
        final_largest_cluster_percentages.append(round(np.mean(largest_cluster_percentages), 3))
        final_std_indexes.append(round(np.std(avg_indexes), 3)/np.sqrt(repeat))
        final_std_percentages.append(round(np.std(largest_cluster_percentages), 3)/np.sqrt(repeat))

    return (final_avg_indexes, final_largest_cluster_percentages,
            final_std_indexes, final_std_percentages)


def plot_props(start, end, step, names: list, indexes: dict, percentages: dict):
    """
    plots the average cluster distance index and the largest cluster percentage for each weight.
    :param start:  the start of the range.
    :param end:  the end of the range.
    :param step:   the step of the range.
    :param names:  the names of the datasets.
    :param indexes:  the average cluster distance indexes.
    :param percentages:  the largest cluster percentages.
    :return:
    """
    # plot the average indexes.
    plt.figure(figsize=(20, 10))
    x_vals = np.linspace(start, end, step)
    for i, name in enumerate(names):
        plt.plot(x_vals, indexes[name][0], label=name, linewidth=5)
        plt.fill_between(x_vals, np.array(indexes[name][0]) - np.array(indexes[name][1]),
                         np.array(indexes[name][0]) + np.array(indexes[name][1]), alpha=0.25)
    plt.xlabel('Exponent (x in $10^{-x}$)')
    plt.ylabel('Average Cluster Distance Index')
    plt.title('Average Cluster Distance Index for Different Weight Proportions')
    plt.legend()
    plt.grid()
    plt.savefig('Figures/weight_proportions_avg_dist.png')
    plt.show()

    # plot the largest cluster percentages.
    plt.figure(figsize=(20, 10))
    for i, name in enumerate(names):
        plt.plot(x_vals, percentages[name][0], label=name, linewidth=5)
        plt.fill_between(x_vals, np.array(percentages[name][0]) - np.array(percentages[name][1]),
                         np.array(percentages[name][0]) + np.array(percentages[name][1]), alpha=0.25)
    plt.xlabel('Exponent (x in $10^{-x}$)')
    plt.ylabel('Largest Cluster Percentage')
    plt.title('Largest Cluster Percentage for Different Weight Proportions')
    plt.legend()
    plt.grid()
    plt.savefig('Figures/weight_proportions_largest_perc.png')
    plt.show()
