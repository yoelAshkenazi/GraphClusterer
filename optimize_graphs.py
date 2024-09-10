import optuna
import networkx as nx
import os
import shutil  # For deleting files
from functions import evaluate_clusters, make_graph, cluster_graph
import pickle as pk
import pandas as pd

# Define ALL_NAMES (list of dataset names)
ALL_NAMES = ['3D printing', "additive manufacturing", "composite material", "autonomous drones", 
             "hypersonic missile", "nuclear reactor", "scramjet", "wind tunnel", 
             "quantum computing", "smart material"]

# Dictionary to store the optimized parameters for each dataset
optimized_params = {}

# Define the objective function to optimize independently for each dataset
def objective(trial, name):
    # Suggest values for weight and resolution for the current dataset
    weight = trial.suggest_float('weight', 0.01, 0.3)
    resolution = trial.suggest_float('resolution', 0.001, 0.2)
    
    # Define the graph creation parameters
    graph_kwargs = {
        'weight': weight,  # Use the optimized weight for this dataset
        'size': 200,       # Default size, adjust as necessary
        'color': '#1f78b4',
        'distance_threshold': 0.5  # Assuming this is fixed for all datasets
    }
    
    # Create the graph for the dataset
    G = make_graph(name, **graph_kwargs)
    
    # Define the clustering parameters, including weight and resolution
    clustering_kwargs = {
        'method': 'louvain',    # Clustering method
        'resolution': resolution,  # Optimized resolution for this dataset
        'weight': weight,        # Optimized weight for clustering
        'save': False            # Save the graph after the best trial, not during optimization
    }
    
    # Cluster the graph with the specified weight and resolution
    clusters = cluster_graph(G, name, **clustering_kwargs)
    
    # Evaluate the clusters
    avg_index, largest_cluster_percentage = evaluate_clusters(G, name)
    
    # Ensure largest_cluster_percentage is between 20% and 60%
    if not (0.2 <= largest_cluster_percentage <= 0.6):
        return float('inf')  # Return a high value to discard this trial
    
    # Report intermediate results for pruning
    trial.report(avg_index, step=0)
    
    # Check if the trial should be pruned
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    # Return the avg_index as the objective to minimize
    return avg_index

# Custom function to save the graph in the 'data/optimized_graphs' directory
def save_graph(G, name, weight, resolution):
    """
    Save the graph in the 'data/optimized_graphs/' directory using the optimized weight and resolution.
    """
    filename = f'data/optimized_graphs/{name}_weight={weight}_resolution={resolution}.gpickle'
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
clear_directory('data/optimized_graphs')

# Run independent optimization for each dataset in ALL_NAMES
for name in ALL_NAMES:
    print(f"Starting optimization for dataset: {name}")
    
    # Create a study for this dataset with a pruning strategy (MedianPruner)
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Optimize the parameters for 2000 trials for the current dataset
    study.optimize(lambda trial: objective(trial, name), n_trials=2000)
    
    # Get the best trial
    best_trial = study.best_trial
    best_weight = best_trial.params['weight']
    best_resolution = best_trial.params['resolution']
    
    # Create the graph using the best parameters
    best_graph_kwargs = {
        'weight': best_weight,
        'size': 200,
        'color': '#1f78b4',
        'distance_threshold': 0.5
    }
    
    # Create the graph using the best parameters
    best_G = make_graph(name, **best_graph_kwargs)
    
    # Use the best resolution and weight for clustering
    best_clustering_kwargs = {
        'method': 'louvain',
        'resolution': best_resolution,
        'weight': best_weight,     # Pass the best weight for clustering
        'save': False
    }
    
    # Cluster the graph with the best resolution and weight
    cluster_graph(best_G, name, **best_clustering_kwargs)
    
    # Evaluate the clusters for the final best graph
    avg_index, largest_cluster_percentage = evaluate_clusters(best_G, name)
    
    # Save the optimized graph directly in 'data/optimized_graphs'
    save_graph(best_G, name, best_weight, best_resolution)
    
    # Store the optimized parameters including avg_index and largest_cluster_percentage
    optimized_params[name] = {
        'weight': best_weight,
        'resolution': best_resolution,
        'avg_index': avg_index,
        'largest_cluster_percentage': largest_cluster_percentage
    }
    
    print(f"Best trial for {name}: avg_index = {avg_index}, "
          f"weight = {best_weight}, resolution = {best_resolution}, "
          f"largest_cluster_percentage = {largest_cluster_percentage}")

# After optimization, print the final table of optimized parameters

# Convert the optimized parameters dictionary to a DataFrame for better display
df = pd.DataFrame(optimized_params).T  # Transpose to make datasets the index
df.reset_index(inplace=True)
df.rename(columns={'index': 'Dataset'}, inplace=True)

# Display the final table
print("\nFinal Optimized Parameters for Each Dataset:\n")
print(df[['Dataset', 'weight', 'resolution', 'avg_index', 'largest_cluster_percentage']])

# Optionally, save the table to a CSV file for future reference
df.to_csv('data/optimized_graphs/optimized_parameters_with_index.csv', index=False)
