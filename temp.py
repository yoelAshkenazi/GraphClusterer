import optuna
import os
from functions import evaluate_clusters, make_graph, cluster_graph, get_file_path, load_pkl
import pickle as pk
import pandas as pd
from calc_energy import compute_energy_distance_matrix
from optuna.samplers import TPESampler

# Define ALL_DATASET_NAMES (list of dataset names)
ALL_DATASET_NAMES = ['3D printing', "additive manufacturing", "composite material", "autonomous drones", "hypersonic missile",
                    "nuclear reactor", "scramjet", "wind tunnel", "quantum computing", "smart material"]

# Dictionary to store the optimized parameters for each dataset
optimized_params = {}

# Track the best non-infinity trial globally
best_valid_trial_params = {}


# Define the objective function to optimize independently for each dataset
def objective(trial, _dataset_name):
    global best_valid_trial_params

    # Suggest values for filter, weight, and resolution for the current dataset
    _least_cutoff = trial.suggest_float('least_cutoff', 0, 5)
    _most_cutoff = trial.suggest_float('most_cutoff', 0, 5)
    _weight_value = trial.suggest_float('weight', 0.01, 0.3)
    _resolution_value = trial.suggest_float('resolution', 0.001, 0.2)

    # Compute the energy distance matrix using the filter value
    _energy_distance_matrix = compute_energy_distance_matrix(_dataset_name, _least_cutoff, _most_cutoff)

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
    print(f"Trial {trial.number} finished with avg_index: {_avg_index:.5f} and largest_cluster_percentage: {_largest_cluster_percentage:.5f}")

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


# Modified clear_directory function
def clear_directory(directory, current_dataset_names):
    """
    Clears obsolete files from the specified directory. Only files that do not correspond
    to the current_dataset_names are deleted. Relevant files are retained and can be overwritten.

    Parameters:
    - directory (str): Path to the directory to clear.
    - current_dataset_names (list): List of dataset names being processed in the current run.
    """
    if os.path.exists(directory):
        existing_files = os.listdir(directory)
        for filename in existing_files:
            # Check if the file corresponds to any dataset in the current list
            # Assuming dataset names in filenames are formatted with underscores instead of spaces
            if not any(dataset.replace(" ", "_") in filename for dataset in current_dataset_names):
                _file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(_file_path):
                        os.unlink(_file_path)
                        print(f"Deleted obsolete file: {_file_path}")
                except Exception as e:
                    print(f"Failed to delete {_file_path}. Reason: {e}")
            else:
                print(f"Retaining existing file: {filename}")
    else:
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")


# Custom function to save the graph
def save_graph(graph, _dataset_name):
    """
    Saves the graph object to a file with a name based on the dataset and parameters.

    Parameters:
    - graph: The graph object to save.
    - _dataset_name (str): Name of the dataset.
    - filter_value (float): Filter parameter used.
    - weight_value (float): Weight parameter used.
    - resolution_value (float): Resolution parameter used.
    """
    # Replace spaces with underscores in dataset name for filename consistency
    formatted_dataset_name = dataset_name.replace(" ", "")
    filename = f'data/optimized_graphs/{formatted_dataset_name}.gpickle'
    try:
        with open(filename, 'wb') as f:
            pk.dump(graph, f, protocol=pk.HIGHEST_PROTOCOL)
        print(f"Graph for '{_dataset_name}' saved successfully to '{filename}'.")
    except Exception as _e:
        print(f"Error saving graph for '{_dataset_name}': {_e}")


# Clear obsolete files in the output directory before starting the optimization
clear_directory('data/optimized_graphs', ALL_DATASET_NAMES)

# Run independent optimization for each dataset in ALL_DATASET_NAMES
for dataset_name in ALL_DATASET_NAMES:
    print(f"\nStarting optimization for dataset: {dataset_name}")

    # Create a TPE sampler with a lower number of startup trials and a higher gamma
    sampler = TPESampler(n_startup_trials=10, gamma=lambda n: min(int(0.1 * n), 25))

    # Create a study for this dataset with a pruning strategy (MedianPruner) and custom TPE Sampler
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        sampler=sampler  # Using custom TPE Sampler
    )

    # Optimize the parameters for 2000 trials for the current dataset
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=2000, show_progress_bar=True, n_jobs=-1)

    # Get the best trial
    best_trial = study.best_trial
    best_least_cutoff = best_trial.params['least_cutoff']
    best_most_cutoff = best_trial.params['most_cutoff']
    best_weight = best_trial.params['weight']
    best_resolution = best_trial.params['resolution']

    # Recompute the energy distance matrix using the best filter value for final evaluation
    energy_distance_matrix = compute_energy_distance_matrix(dataset_name, best_least_cutoff, best_most_cutoff)

    # Load paper IDs again to ensure they are available for final evaluation
    file_path = get_file_path(dataset_name)
    embeddings = load_pkl(file_path)
    paper_ids = embeddings['IDs']

    # Create the graph using the best parameters and ensure the distance_matrix is passed
    best_graph_kwargs = {
        'weight': best_weight,
        'size': 200,
        'color': '#1f78b4',
        'distance_threshold': 0.5,
        'use_only_distances': True,
        'use_only_original': True,
        'distance_matrix': energy_distance_matrix  # Include the recomputed distance matrix here
    }

    best_graph = make_graph(dataset_name, **best_graph_kwargs)

    # Use the best resolution and weight for clustering
    best_clustering_kwargs = {
        'method': 'louvain',
        'resolution': best_resolution,
        'weight': best_weight,
        'save': False
    }

    cluster_graph(best_graph, dataset_name, **best_clustering_kwargs)

    avg_index, largest_cluster_percentage = evaluate_clusters(best_graph, dataset_name)

    # Print the best trial information, only printing avg_index
    print(f"Best trial is {best_trial.number} with avg_index: {study.best_value:.5f} and largest_cluster_percentage: {largest_cluster_percentage:.5f}")

    # Save the optimized graph
    save_graph(best_graph, dataset_name)

    # Store the optimized parameters
    optimized_params[dataset_name] = {
        'least_cutoff': best_least_cutoff,
        'most_cutoff': best_most_cutoff,
        'weight': best_weight,
        'resolution': best_resolution,
        'avg_index': avg_index,
        'largest_cluster_percentage': largest_cluster_percentage
    }

    print(f"Best trial for {dataset_name}: avg_index = {avg_index}, largest_cluster_percentage = {largest_cluster_percentage:.5f}")

# Convert the optimized parameters dictionary to a DataFrame
df = pd.DataFrame(optimized_params).T
df.reset_index(inplace=True)
df.rename(columns={'index': 'Dataset'}, inplace=True)

# Display the final table
print("\nFinal Optimized Parameters for Each Dataset:\n")
print(df[['Dataset', 'least_cutoff', 'most_cutoff', 'weight', 'resolution', 'avg_index', 'largest_cluster_percentage']])

# Optionally, save the table to a CSV file for future reference
output_csv_path = 'data/optimized_graphs/optimized_parameters_with_filter.csv'
try:
    df.to_csv(output_csv_path, index=False)
    print(f"\nOptimized parameters saved successfully to '{output_csv_path}'.")
except Exception as e:
    print(f"Error saving optimized parameters to CSV: {e}")