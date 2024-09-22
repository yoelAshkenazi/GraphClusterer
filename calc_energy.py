import os
from os.path import join
import pickle
import numpy as np
import dcor
from scipy.stats import gaussian_kde
from copy import deepcopy


def filter_data(combined_list, x):
    """
    Filters the combined data list by removing a percentage of data from both ends of the distribution and from the peak.
    This is done by applying a Gaussian Kernel Density Estimate (KDE) to identify and remove data around the peak.
    
    :param combined_list: List of combined embeddings
    :param x: Percentage of data to remove
    :return: filtered list of data, removed elements, KDE x and KDE values
    """
    # Sort the data
    combined_list_sorted = np.sort(combined_list)
    n = len(combined_list_sorted)
    
    # Calculate the indices to remove for left, right, and peak cutoffs
    left_cutoff = int((x / 4) * n / 100)
    right_cutoff = int(n - (x / 4) * n / 100)
    peak_cutoff_count = int((x / 2) * n / 100)

    # Remove the data from the left and right sides
    left_removed = combined_list_sorted[:left_cutoff]
    right_removed = combined_list_sorted[right_cutoff:]
    
    # Keep the data within the range
    filtered_list = combined_list_sorted[left_cutoff:right_cutoff]

    # Apply Kernel Density Estimation (KDE) to the filtered list
    kde = gaussian_kde(filtered_list, bw_method='silverman')
    
    # Generate x values for the KDE plot
    x_vals = np.linspace(min(filtered_list), max(filtered_list), 1000)
    
    # Calculate KDE values for the given x_vals
    kde_vals = kde(x_vals)

    # Find the peak of the KDE
    peak_index = np.argmax(kde_vals)
    peak_value = x_vals[peak_index]

    # Calculate distances from the peak value and identify elements to remove
    distances_from_peak = np.abs(filtered_list - peak_value)
    peak_indices_to_remove = np.argsort(distances_from_peak)[:peak_cutoff_count]
    peak_removed = filtered_list[peak_indices_to_remove]

    # Remove the peak elements from the filtered list
    filtered_list_final = np.delete(filtered_list, peak_indices_to_remove)
    
    # Combine all removed elements (left, right, and peak)
    removed_elements_combined = np.concatenate([left_removed, right_removed, peak_removed])

    return filtered_list_final, removed_elements_combined, x_vals, kde_vals


def compute_energy_distance_matrix(ds_name, x):
    """
    Compute the energy distance matrix for the embedding file that starts with 'ds_name'.
    
    :param ds_name: The dataset name to look for.
    :param x: Percentage of data to remove.
    :return: The energy distance matrix for the matched embedding.
    """
    EMBEDDINGS_DIR = 'data/embeddings'
    
    # Find the embedding file that starts with 'ds_name'
    embedding_path = None
    for _embedding in os.listdir(EMBEDDINGS_DIR):
        if _embedding.startswith(ds_name):
            embedding_path = os.path.join(EMBEDDINGS_DIR, _embedding)
            break

    # If no file was found that starts with 'ds_name', raise an error
    if embedding_path is None:
        raise FileNotFoundError(f"No embedding file found starting with '{ds_name}' in {EMBEDDINGS_DIR}")

    # Load the embeddings from the file
    with open(embedding_path, 'rb') as f:
        embedding_dct = pickle.load(f)

    # Extract the list of IDs (keys) from the embedding dictionary
    ids = list(embedding_dct.keys())
    combined_list = []

    # Combine all embedding arrays into a single list for filtering
    for id_ in ids:
        for embedding_array in embedding_dct[id_]:
            combined_list.extend(embedding_array.tolist())

    # Filter the data and remove outliers from the combined list
    filtered_list_final, removed_elements_combined, x_vals, kde_vals = filter_data(combined_list, x)

    # Create a deep copy of the original embedding dictionary for modification
    temp_dict = deepcopy(embedding_dct)

    # Iterate through each key-value pair in the embedding dictionary
    for key, value in embedding_dct.items():
        # Remove values that were identified as outliers
        filtered_values = [v for v in value if v.tolist() not in removed_elements_combined.tolist()]
        temp_dict[key] = np.array(filtered_values)

    # Initialize an empty matrix for the energy distances
    energy_distance_matrix = np.zeros(shape=(len(ids), len(ids)))

    # Compute the energy distance for each pair of embeddings
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_i = ids[i]
            id_j = ids[j]
            embedding_i = temp_dict[id_i]
            embedding_j = temp_dict[id_j]

            # Calculate energy distance using dcor package
            energy_distance = dcor.energy_distance(embedding_i, embedding_j)

            # Symmetrically assign the distance to the matrix
            energy_distance_matrix[i, j] = energy_distance
            energy_distance_matrix[j, i] = energy_distance

    # Return just the energy distance matrix
    return energy_distance_matrix
