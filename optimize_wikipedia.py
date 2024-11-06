import optuna
import os
import pickle as pk
import pandas as pd
import random
import networkx as nx

# Define the list of dataset names to optimize
"""ALL_DATASET_NAMES = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]
"""
ALL_DATASET_NAMES = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]
# Dictionary to store the optimized parameters for each dataset
optimized_params = {}

# Track the best non-infinity trial globally
best_valid_trial_params = {}

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


def save_graph(G, name, resolution):
    filename = f'data/wikipedia_optimized/{name}_resolution={resolution}.gpickle'
    # Dump the graph to a .gpickle file.
    try:
        with open(filename, 'wb') as f:
            pk.dump(G, f, protocol=pk.HIGHEST_PROTOCOL)
        print(f"Graph for '{name}' saved successfully to '{filename}'.")
    except Exception as e:
        print(f"Error saving graph for '{name}': {e}")


# Function to clear the directory if it exists and contains files
def clear_directory(directory):
    if os.path.exists(directory):
        # Check if directory is not empty
        if os.listdir(directory):
            print(f"Directory '{directory}' is not empty. Deleting existing files...")
            # Remove all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Remove the file
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory '{directory}' is empty. Proceeding with file creation.")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")


# Clear the output directory before starting the optimization
clear_directory('data\wikipedia_optimized')


# Define the objective function to optimize independently for each dataset
def objective(G, name, trial):
    # Suggest values for weight and resolution for the current dataset
    resolution = trial.suggest_float('resolution', 0.1, 0.3) if name == 'London' else trial.suggest_float('resolution', 0.1, 0.2)
        
    # Cluster the graph with the specified weight and resolution
    clusters = cluster_graph(G, resolution)
    
    # Evaluate the clusters
    largest_cluster_percentage = evaluate_clusters(G)
    print(largest_cluster_percentage)
    # Ensure largest_cluster_percentage is between 20% and 75%
    if not (0.2 <= largest_cluster_percentage <= 0.75):
        return float('inf')  # Return a high value to discard this trial
    
    # Report intermediate results for pruning
    trial.report(largest_cluster_percentage, step=0)
    
    # Check if the trial should be pruned
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    # Return the avg_index as the objective to minimize
    return largest_cluster_percentage

# Run independent optimization for each dataset in ALL_NAMES
for name in ALL_DATASET_NAMES:
    print(f"Starting optimization for dataset: {name}")
    
    G = load_graph(name)
    # Create a study for this dataset with a pruning strategy (MedianPruner)
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Optimize the parameters for 2000 trials for the current dataset
    study.optimize(lambda trial: objective(G, name, trial), n_trials=2000, show_progress_bar=True)


    # Get the best trial
    best_trial = study.best_trial
    best_resolution = best_trial.params['resolution']
    clusters = cluster_graph(G, best_resolution)

    best_largest_cluster_percentage = evaluate_clusters(G)
    
    # Store the optimized parameters in the dictionary
    optimized_params[name] = {'resolution': best_resolution, 'largest_cluster_percentage': best_trial.value}
    
    print(f"\nBest trial for {name}: largest_cluster_percentage = {best_trial.value}, resolution = {best_resolution}")
    save_graph(G, name, best_resolution)


# Convert the optimized parameters dictionary to a DataFrame for better display
df = pd.DataFrame(optimized_params).T  # Transpose to make datasets the index
print("\nFinal Optimized Parameters for Each Dataset:\n")
print(df)
# Optionally, save the table to a CSV file for future reference
df.to_csv('data/wikipedia_optimized/optimized_parameters.csv')