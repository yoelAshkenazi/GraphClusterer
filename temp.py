import networkx as nx
import matplotlib.pyplot as plt
import functions
import random

print_info = True
# set the parameters for the clustering.
clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': 0.15, 'use_only_distances': False, 'weight': 10 ** (-0.7),
                         'use_only_original': True, 'proportion': 0.5, 'K': 5}

def draw_graph(G, name, **kwargs):
    """
    Draw a graph with the given colors and shapes.
    :param G: the graph to draw.
    :param name: the name of the graph.
    :return:
    """
    functions.cluster_graph(G, name, **clustering_kwargs)
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

    blue_weights = [G[u][v].get('weight', 1) for u, v in G.edges() if G[u][v]['color'] == 'blue']  # weights for blue edges.
    red_weights = [15 * G[u][v].get('weight', 1) if G[u][v].get('weight', 1) != 1 else 1 for u, v in G.edges() if G[u][v]['color'] == 'red']  # weights for red edges.
    # draw vertices with given shapes and sizes.
    shapes = nx.get_node_attributes(G, 'shape').values()
    # draw the vertices according to shapes.
    for shape in shapes:
        # get a list of nodes with the same shape.
        vertices_ = [v for v in G.nodes() if
                    shape == G.nodes()[v]['shape']]
        # get a list of the sizes of said nodes.
        vertex_sizes = [G.nodes()[node].get('size', 500) for node in vertices_]
        # get a list of the colors of said nodes.
        vertex_colors = [G.nodes[node]['color'] for node in vertices_]
        nx.draw_networkx_nodes(G, pos, nodelist=vertices_, node_size=vertex_sizes,
                            node_shape=shape, node_color=vertex_colors, alpha=0.8,
                            linewidths=1.5, edgecolors='black')  # always draw a black border.

    # draw the blue edges.
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'blue'],
                        edge_color='blue', width=blue_weights, alpha=0.75)
    # draw the red edges. (with a weight of 10 times the original weight)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if G[u][v]['color'] == 'red'],
                        edge_color='red', width=red_weights, alpha=0.75)
    plt.show()