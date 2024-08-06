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


def load_graph(name: str, version: str, proportion, k: int = 5, weight: float = 1) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :param version: the version of the graph.
    :param k: the KNN parameter.
    :param proportion: the proportion of the graph.
    :param weight: the weight of the edges.
    :return:
    """

    assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', " \
                                                               "or 'proportion'."

    graph_path = f"data/processed_graphs/k_{k}/"
    if weight != 1:
        graph_path += f"weight_{weight}/"
    if version == 'distances':
        graph_path += f"only_distances/{name}.gpickle"

    elif version == 'original':
        graph_path += f"only_original/{name}.gpickle"

    else:
        if proportion != 0.5:
            graph_path += f"{name}_proportion_{proportion}.gpickle"
        else:
            graph_path += f"{name}.gpickle"

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


def make_graph(name, **kwargs):
    """
    Create a graph with the given parameters.
    :param name: file name.
    :param kwargs: additional arguments for the graph.
    :return:

    ------------
    Example:
    ------------
    name = '3D printing'

    graph_kwargs = {'A': 1.0, 'size': 200, 'color': '#1f78b4', 'distance_threshold': 0.5}

    G = functions.make_graph('3D printing', **graph_kwargs)

    this will create a graph for the '3D printing' embeddings, with the given vertex sizes and colors.
    """
    # load the embeddings.
    file_path = get_file_path(name)
    embeddings = load_pkl(file_path)
    dists = embeddings['Distances']  # distances between papers.
    paper_ids = embeddings['IDs']  # paper ids.

    # set the parameters.
    A = kwargs['A'] if 'A' in kwargs else 1.0
    K = kwargs['K'] if 'K' in kwargs else None
    default_vertex_color = kwargs['color'] if 'color' in kwargs else '#1f78b4'
    default_size = kwargs['size'] if 'size' in kwargs else 150
    threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
    use_distances = kwargs.get('use_only_distances', True)  # whether to use only the distances.
    use_original = kwargs.get('use_only_original', True)  # whether to use only the original edges.
    proportion = kwargs.get('proportion', 0.5)  # the proportion of the original edge weights to use.
    weight = kwargs.get('weight', 1)  # the weight of the edges.
    # assign shapes to each type.
    shapes = {'paper': 'o', 'author': '*', 'keyword': 'd', 'institution': 'p', 'country': 's'}

    # create a graph.
    G = nx.Graph()
    for i, paper_id in enumerate(paper_ids):  # add the papers to the graph.
        G.add_node(paper_id, size=default_size, shape='o', type='paper', color=default_vertex_color)

    # add the other types of vertices to the graph, as well as the blue edges.
    df = pd.read_csv('data/graphs/' + name + '_graph.csv')
    targets = df['target']
    types = df['type']
    ids = df['paper_id']

    # add the vertices to the graph.
    if use_original:
        for j, target in enumerate(targets):
            if target == '':  # skip empty targets.
                continue
            G.add_node(target, size=default_size, shape=shapes[types[j]], type=types[j], color=default_vertex_color)

        for j in range(len(targets)):
            if weight == 1:
                G.add_edge(ids[j], targets[j], weight=(2 * A * proportion), color='blue')
            else:
                G.add_edge(ids[j], targets[j], weight=1, color='blue')
            # add the blue edges with a weight of A.

    # add the red edges to the graph.
    if use_distances:
        for i in range(len(paper_ids)):
            if K is not None:
                # use KNN to add the red edges.
                # get the K nearest neighbors.
                indices = dists[i].argsort()[1: K + 1]  # get the K nearest neighbors not including the paper itself.
                for j in indices:
                    if weight == 1:
                        G.add_edge(paper_ids[i], paper_ids[j], weight=(dists[i, j] * (1 - proportion)), color='red')
                    else:
                        G.add_edge(paper_ids[i], paper_ids[j], weight=weight, color='red')
                    # add the red edges.
                continue

            for j in range(i + 1, len(paper_ids)):
                if dists[i, j] > threshold:  # skip the zero distances.
                    continue

                if weight == 1:
                    G.add_edge(paper_ids[i], paper_ids[j], weight=(dists[i, j] * (1 - proportion)), color='red')
                else:
                    G.add_edge(paper_ids[i], paper_ids[j], weight=weight, color='red')
                # add the red edges.

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

    rate = kwargs['shown_percentage'] if 'shown_percentage' in kwargs else 0.2
    figsize = kwargs['figsize'] if 'figsize' in kwargs else 20
    save = kwargs['save'] if 'save' in kwargs else False
    method = kwargs['method'] if 'method' in kwargs else 'louvain'
    use_original = kwargs.get('use_only_original', True)  # whether to use only the original edges.
    use_distances = kwargs.get('use_only_distances', True)  # whether to use only the distances.
    proportion = kwargs.get('proportion', 0.5)  # the proportion of the original edge weights to use.

    plt.figure(figsize=(figsize, figsize))
    pos = nx.spring_layout(G)  # positions for all nodes.

    # randomly select a rate of vertices to draw.
    vertices = random.sample(list(G.nodes), int(rate * len(G.nodes())))
    G = G.subgraph(vertices)

    blue_weights = [G[u][v]['weight'] for u, v in G.edges() if G[u][v]['color'] == 'blue']  # weights for blue edges.
    red_weights = [15 * G[u][v]['weight'] for u, v in G.edges() if G[u][v]['color'] == 'red']  # weights for red edges.
    # draw vertices with given shapes and sizes.
    shapes = nx.get_node_attributes(G, 'shape').values()
    # draw the vertices according to shapes.
    for shape in shapes:
        # get a list of nodes with the same shape.
        vertices_ = [v for v in G.nodes() if
                     shape == G.nodes.data()[v]['shape']]
        # get a list of the sizes of said nodes.
        vertex_sizes = [G.nodes.data()[node]['size'] for node in vertices_]
        # get a list of the colors of said nodes.
        vertex_colors = [G.nodes.data()[node]['color'] for node in vertices_]
        nx.draw_networkx_nodes(G, pos, nodelist=vertices_, node_size=vertex_sizes,
                               node_shape=shape, node_color=vertex_colors, alpha=0.8,
                               linewidths=1.5, edgecolors='black')  # always draw a black border.

    # draw the blue edges.
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'blue'],
                           edge_color='blue', width=blue_weights, alpha=0.75)
    # draw the red edges. (with a weight of 10 times the original weight)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'red'],
                           edge_color='red', width=red_weights, alpha=0.75)

    if save:
        filename = f'Figures/{int(100 * rate)}_percents_shown/{method}_method/{name}'
        if not use_original:
            filename += '_only_distances'
        elif not use_distances:
            filename += '_only_original'
        if proportion != 0.5:
            filename += f'_proportion_{proportion}'
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
    # get the clustering method.
    save = kwargs['save'] if 'save' in kwargs else False
    method = kwargs['method'] if 'method' in kwargs else 'louvain'
    use_original = kwargs.get('use_only_original', True)  # whether to use only the original edges.
    use_distances = kwargs.get('use_only_distances', True)  # whether to use only the distances.
    proportion = kwargs.get('proportion', 0.5)  # the proportion of the original edge weights to use.
    K = kwargs.get('K', 5)  # the KNN parameter.
    weight = kwargs.get('weight', 1)  # the weight of the edges.

    if method == 'louvain':  # use the louvain method.
        res = kwargs['resolution'] if 'resolution' in kwargs else 1.0
        partition = nx.algorithms.community.louvain_communities(G, resolution=res)
    elif method == 'leiden':  # use the leiden method.
        res = kwargs['resolution'] if 'resolution' in kwargs else 1.0
        partition = nx.algorithms.community.quality.modularity(G, resolution=res)
    elif method == 'k_clique':  # use the k-clique method.
        k = kwargs['k'] if 'k' in kwargs else 5
        partition = nx.algorithms.community.k_clique_communities(G, k)
    else:
        raise ValueError("Invalid clustering method.")

    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(partition))]
    for i, cluster in enumerate(partition):
        for node in cluster:
            G.nodes[node]['color'] = colors[i]

    if save:  # save the graph.
        dirname = f'data/processed_graphs/k_{K}/'
        if weight != 1:
            dirname += f'weight_{weight}/'
        if not use_original:
            dirname += f'only_distances/'
        elif not use_distances:
            dirname += f'only_original/'
        filename = dirname + name
        if proportion != 0.5:
            filename += f'_proportion_{proportion}'
        filename += '.gpickle'
        # dump the graph to a .pkl file.
        try:
            with open(filename, 'wb') as f:
                pk.dump(G, f, protocol=pk.HIGHEST_PROTOCOL)
            print(f"Graph for '{name}' saved successfully to '{filename}'.")
        except FileNotFoundError:  # create the directory if it doesn't exist.
            import os
            os.makedirs(dirname)
            with open(filename, 'wb') as f:
                pk.dump(G, f, protocol=pk.HIGHEST_PROTOCOL)
            print(f"Graph for '{name}' saved successfully to '{filename}'.")

    return partition


def analyze_clusters(G):
    """
    Analyze the clusters of the graph.
    :param G: the graph.
    :return: a dictionary with the amount of papers in each cluster.
    """
    # first we filter articles by vertex type.
    articles = [node for node in G.nodes() if G.nodes.data()[node]['type'] == 'paper']
    articles_graph = G.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    nodes = graph.nodes(data=True)
    colors = set([node[1]['color'] for node in nodes])
    sizes = []
    for color in colors:  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]
        sizes.append(len(nodes))

    return sizes


def evaluate_clusters(G, name):
    """
    Evaluate the clusters of the graph.
    :param name: the name of the dataset.
    :param G:  the graph.
    :return:
    """

    # load the embeddings.
    file_path = get_file_path(name)
    embeddings = load_pkl(file_path)
    dists = embeddings['Distances']  # distances between papers.
    paper_ids = embeddings['IDs']  # paper ids.

    # get the total average distance.
    avg_all_dists = 0
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            avg_all_dists += dists[i, j]
    avg_all_dists /= (len(paper_ids) * (len(paper_ids) - 1) / 2)

    # get the average distance for each cluster.
    # filter the vertices by color and type.
    articles = [node for node in G.nodes() if G.nodes.data()[node]['type'] == 'paper']
    articles_graph = G.subgraph(articles)
    colors = set([node[1]['color'] for node in articles_graph.nodes(data=True)])
    sizes = []
    avg_cluster_dists = []
    for color in colors:
        cluster = [node for node in articles_graph.nodes() if articles_graph.nodes.data()[node]['color'] == color]
        sizes.append(len(cluster))
        avg_cluster_dist = 0
        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                if paper_ids[i] in cluster and paper_ids[j] in cluster:
                    avg_cluster_dist += dists[i, j]
        if len(cluster) == 1:
            avg_cluster_dist = 0
        else:
            avg_cluster_dist /= (len(cluster) * (len(cluster) - 1) / 2)
        avg_cluster_dists.append(avg_cluster_dist)

    # get the average distance for the clusters.
    avg_cluster_dists = sum(avg_cluster_dists) / len(avg_cluster_dists)
    avg_index = avg_cluster_dists / avg_all_dists  # get the average index.
    avg_index = round(avg_index, 5)
    # get the percentage of papers in the largest cluster.
    largest_cluster = max(sizes)
    largest_cluster_percentage = largest_cluster / len(articles)

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
            avg_index, largest_cluster_percentage = evaluate_clusters(G_copy, name)
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
