import optuna
import os
from functions import evaluate_clusters, make_graph, cluster_graph, get_file_path, load_pkl
import pickle as pk
import pandas as pd
from calc_energy import compute_energy_distance_matrix
from optuna.samplers import TPESampler
import random

ALL_DATASET_NAMES = ['3D printing']

# Dictionary to store the optimized parameters for each dataset
optimized_params = {}

# Track the best non-infinity trial globally
best_valid_trial_params = {}

# Define the objective function to optimize independently for each dataset
def objective(_dataset_name, counter):
    global best_valid_trial_params

    # Suggest values for filter, weight, and resolution for the current dataset
    _least_cutoff = random.uniform(0, 5)
    _most_cutoff = random.uniform(0, 5)
    _weight_value = random.uniform(0.01, 0.3)
    _resolution_value = random.uniform(0.001, 0.2)

    print(f"Trial {counter} for {_dataset_name} with least_cutoff: {_least_cutoff:.5f}, most_cutoff: {_most_cutoff:.5f}, weight: {_weight_value:.5f}, resolution: {_resolution_value:.5f}")
    print("Creating energy distance matrix...")

    # Generate a unique KDE file path for the dataset
    kde_file = f"kde/{_dataset_name}_kde_values.pkl"

    # Compute the energy distance matrix using the filter value
    _energy_distance_matrix = compute_energy_distance_matrix(
        _dataset_name, _least_cutoff, _most_cutoff, kde_file
    )

    # Define the graph creation parameters using the computed matrix and optimized weight
    graph_kwargs = {
        'weight': _weight_value,
        'size': 200,
        'color': '#1f78b4',
        'distance_threshold': 0.5,
        'use_only_distances': True,
        'use_only_original': True,
        'distance_matrix': _energy_distance_matrix
    }

    # Create the graph for the dataset
    graph = make_graph(_dataset_name, **graph_kwargs)

    # Define the clustering parameters, including weight and resolution
    clustering_kwargs = {
        'method': 'louvain',
        'resolution': _resolution_value,
        'weight': _weight_value,
        'save': False
    }

    # Cluster the graph with the specified weight and resolution
    cluster_graph(graph, _dataset_name, **clustering_kwargs)

    # Evaluate the clusters using only the graph and dataset name
    _avg_index, _largest_cluster_percentage = evaluate_clusters(graph, _dataset_name)

    # Ensure largest_cluster_percentage is between 20% and 75%
    if not (0.2 <= _largest_cluster_percentage <= 0.75):
        print(f"Trial pruned: largest_cluster_percentage {_largest_cluster_percentage} out of range.")
        return float('inf')  # Return a high value to discard this trial

    # Print results for the current trial
    print(f"Trial {counter} finished with avg_index: {_avg_index:.5f} and largest_cluster_percentage: {_largest_cluster_percentage:.5f}")

    # If this trial is valid, store it as the best valid trial
    if _dataset_name not in best_valid_trial_params or _avg_index < best_valid_trial_params[_dataset_name]['avg_index']:
        print(f"New best trial for {_dataset_name}: avg_index = {_avg_index:.5f}")
        best_valid_trial_params[_dataset_name] = {
            'least_cutoff': _least_cutoff,
            'most_cutoff': _most_cutoff,
            'weight': _weight_value,
            'resolution': _resolution_value,
            'avg_index': _avg_index,
            'largest_cluster_percentage': _largest_cluster_percentage  # Keeping this internally for logic purposes
        }

    # Return avg_index as the objective to minimize
    return _avg_index


for name in ALL_DATASET_NAMES:
    for i in range(1, 51):
        counter = i
        avg_index = objective(name, counter)
        print(avg_index)
