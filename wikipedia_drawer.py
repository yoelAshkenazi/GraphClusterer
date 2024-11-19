import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle as pk
import pandas as pd

"""
ALL_DATASET_NAMES = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]
"""
ALL_DATASET_NAMES = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]

def draw_graph(G, name, **kwargs):
    """
    Draw a graph with the given colors and shapes.
    :param G: the graph to draw.
    :param name: the name of the graph.
    :return:
    """

    rate = 1
    figsize = kwargs['figsize'] if 'figsize' in kwargs else 10
    save = kwargs['save'] if 'save' in kwargs else False
    method = kwargs['method'] if 'method' in kwargs else 'louvain'

    plt.figure(figsize=(figsize, figsize))
    pos = nx.spring_layout(G)  # positions for all nodes.

    # randomly select a rate of vertices to draw.
    vertices = random.sample(list(G.nodes), int(rate * len(G.nodes())))
    G = G.subgraph(vertices)

    red_weights = [5 for u, v in G.edges() if G[u][v]['color'] == 'red']  # weights for red edges.
    # draw vertices with given shapes and sizes.
    shapes = nx.get_node_attributes(G, 'shape').values()
    # draw the vertices according to shapes.
    for shape in shapes:
        # get a list of nodes with the same shape.
        vertices_ = [v for v in G.nodes() if
                     shape == G.nodes.data()[v]['shape']]
        # get a list of the sizes of said nodes.
        vertex_sizes = [500 for node in vertices_]
        # get a list of the colors of said nodes.
        vertex_colors = [G.nodes.data()[node]['color'] for node in vertices_]
        nx.draw_networkx_nodes(G, pos, nodelist=vertices_, node_size=vertex_sizes,
                               node_shape=shape, node_color=vertex_colors, alpha=0.8,
                               linewidths=1.5, edgecolors='black')  # always draw a black border.

    # draw the red edges. (with a weight of 10 times the original weight)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'red'],
                           edge_color='red', width=red_weights, alpha=0.75)

    if save:
        filename = f'Figures/wikipedia/{name}'
        try:
            plt.savefig(f'{filename}.png')
        except FileNotFoundError:  # create the directory if it doesn't exist.
            import os
            os.makedirs(f'Figures/{int(100 * rate)}_percents_shown/{method}_method/')
            plt.savefig(f'{filename}.png')
    plt.show()

def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return:
    """

    graph_path = None
    for file in os.listdir('data/wikipedia_optimized'):
        if file.startswith(name):
            graph_path = 'data/wikipedia_optimized/' + file

    # load the graph.

    with open(graph_path, 'rb') as f:
        graph = pk.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    return graph

draw_kwargs = {
        'save': True,
        'method': 'louvain',
        'shown_percentage': 0.3
    }
for name in ALL_DATASET_NAMES:
    G = load_graph(name)
    draw_graph(G, name, **draw_kwargs)
