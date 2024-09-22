import os
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl
import random
import pandas as pd
import functions


print_info = True
# set the parameters for the clustering.
clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': 0.15, 'use_only_distances': False, 'weight': 10 ** (-0.7),
                         'use_only_original': True, 'proportion': 0.5, 'K': 5}


def find_files_by_name(name):
    directory='data/optimized_graphs'
    matching_files = []
    for filename in os.listdir(directory):
        if filename.startswith(name):
            matching_files.append(filename)
            break
    return matching_files

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

def load_wikipedia_graph(name: str) -> nx.Graph:
    """
    Load the graph whose file name starts with the given name.
    :param name: The base name of the graph to load (e.g., "3D printing").
    :return: A filtered NetworkX graph object.
    """

    # Define the base directory where the graphs are stored.
    base_dir = 'data/wikipedia_graphs/'

    # Get the list of files in the directory.
    graph_files = os.listdir(base_dir)
    
    # Find the file that starts with the given name.
    matching_file = None
    for file in graph_files:
        if file.startswith(name):
            matching_file = file
            break
    
    if not matching_file:
        raise FileNotFoundError(f"No graph file found for {name} in {base_dir}")
    
    # Build the full file path to the graph file.
    graph_path = os.path.join(base_dir, matching_file)
    
    # Load the graph from the found file.
    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # Filter the graph to remove NaN nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    
    # Return the subgraph containing only the non-NaN nodes.
    graph = graph.subgraph(nodes)
    
    return graph

ALL_NAMES = ['apple', "car", "clock", "house", "London", "mathematics", "snake", "turtle"]
for name in ALL_NAMES:
    G = load_wikipedia_graph(name)

    draw_kwargs = {'shown_percentage': 0.65, 'figsize': 10, 'save': False, 'method': 'louvain'}
    draw_graph(G, name, **draw_kwargs)

