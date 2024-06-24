"""
Yoel Ashkenazi
Clustering the graphs using the original edges and similarity edges based on the distances.
"""

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
    default_vertex_color = kwargs['color'] if 'color' in kwargs else '#1f78b4'
    default_size = kwargs['size'] if 'size' in kwargs else 150
    threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
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
    for j, target in enumerate(targets):
        if target == '':  # skip empty targets.
            continue
        G.add_node(target, size=default_size, shape=shapes[types[j]], type=types[j], color=default_vertex_color)

    # add the blue edges to the graph.
    for j in range(len(targets)):
        G.add_edge(ids[j], targets[j], weight=2 * A, color='blue')  # add the blue edges with a weight of A.

    # add the red edges to the graph.
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            if dists[i, j] > threshold:  # skip the zero distances.
                continue

            G.add_edge(paper_ids[i], paper_ids[j], weight=A + 10 * dists[i, j], color='red')  # add the red edges.

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

    plt.figure(figsize=(figsize, figsize))
    pos = nx.spring_layout(G)  # positions for all nodes.

    # randomly select a rate of vertices to draw.
    vertices = random.sample(list(G.nodes), int(rate * len(G.nodes())))
    G = G.subgraph(vertices)

    colors = [G[u][v]['color'] for u, v in G.edges()]  # colors for all edges.
    weights = [G[u][v]['weight'] for u, v in G.edges()]  # weights for all edges.
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

    # draw the edges.
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=colors, width=weights)
    if save:
        try:
            plt.savefig(f'Figures/{int(100 * rate)}_percents_shown/{method}_method/{name}.png')
        except FileNotFoundError:  # create the directory if it doesn't exist.
            import os
            os.makedirs(f'Figures/{int(100 * rate)}_percents_shown/{method}_method/')
            plt.savefig(f'Figures/{int(100 * rate)}_percents_shown/{method}_method/{name}.png')
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

    if save:
        # dump the graph to a .pkl file.
        with open(f'data/processed_graphs/{name}.pkl', 'wb') as f:
            pk.dump(G, f)
            print(f"Graph for '{name}' saved successfully to 'data/processed_graphs/{name}.pkl'.")

    return partition
