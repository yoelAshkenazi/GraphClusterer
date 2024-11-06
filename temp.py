import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import networkx as nx
import os
import pickle as pk
import pandas as pd

def load_graph(name):
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return: the graph.
    """
    graph_path = None
    for file in os.listdir('data/wikipedia_graphs'):
        if file.startswith(name):
            graph_path = 'data/wikipedia_graphs/' + file

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

def cluster_graph(G, res):
    """
    divide the vertices of the graph into clusters.
    set a random color to each cluster's nodes.
    :param G: the graph to cluster.
    :param kwargs: additional arguments for the clustering method.
    :return:
    """

    partition = nx.algorithms.community.louvain_communities(G, resolution=res)

    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(partition))]
    for i, cluster in enumerate(partition):
        for node in cluster:
            G.nodes[node]['color'] = colors[i]

    return partition

def evaluate_clusters(G):
    """
    Evaluate the clusters of the graph.
    :param G:  the graph.
    :return: largest_cluster_percentage
    """
    
    # filter the vertices by type 'paper'.
    articles = [node for node in G.nodes() if G.nodes.data()[node].get('type', 'paper')]
    articles_graph = G.subgraph(articles)
    colors = set([node[1]['color'] for node in articles_graph.nodes(data=True)])
    
    # Calculate the size of each cluster.
    sizes = []
    for color in colors:
        cluster = [node for node in articles_graph.nodes() if articles_graph.nodes.data()[node]['color'] == color]
        sizes.append(len(cluster))
    
    # Calculate the percentage of papers in the largest cluster.
    largest_cluster = max(sizes)
    largest_cluster_percentage = largest_cluster / len(articles)
    
    return largest_cluster_percentage


def main():
    name = "apple"
    G = load_graph(name)
    
    # Define the range and number of resolution values
    resolution_start = 0.001
    resolution_end = 1.0
    num_resolutions = 500  # Number of distinct resolution values
    trials_per_resolution = 1000  # Number of trials per resolution
    
    # Generate a list of resolution values
    resolution_values = np.linspace(resolution_start, resolution_end, num_resolutions)
    
    # Lists to store aggregated results
    avg_largest_cluster = []
    counter = 0  # To count trials where 0.6 <= percentage < 0.75
    
    total_iterations = num_resolutions * trials_per_resolution
    
    # Initialize tqdm progress bar
    with tqdm(total=total_iterations, desc="Processing", unit="trial") as pbar:
        for resolution in resolution_values:
            cluster_percentages = []
            for _ in range(trials_per_resolution):
                clusters = cluster_graph(G, resolution)
                largest_cluster_percentage = evaluate_clusters(G)
                cluster_percentages.append(largest_cluster_percentage)
                
                if 0.6 <= largest_cluster_percentage < 0.75:
                    counter += 1
                
                pbar.update(1)
            
            # Compute average for current resolution
            avg = np.mean(cluster_percentages)
            avg_largest_cluster.append(avg)
    
    # Compute overall statistics
    total_trials = total_iterations
    percentage_counter = counter / total_trials
    
    print(f"\nMax Average: {max(avg_largest_cluster):.4f}, Min Average: {min(avg_largest_cluster):.4f}")
    print(f"% of trials with 0.6 <= largest_cluster_percentage < 0.75: {percentage_counter:.2%}")
    
    # Plotting Section
    plt.figure(figsize=(12, 7))
    plt.plot(resolution_values, avg_largest_cluster, label='Average Largest Cluster Percentage', color='blue', linewidth=2)
    
    plt.xlabel('Resolution')
    plt.ylabel('Average Largest Cluster Percentage')
    plt.title('Average Largest Cluster Percentage vs. Resolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
