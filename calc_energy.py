import os
import pickle
import numpy as np
import dcor
from copy import deepcopy
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA


def custom_kde(data, bandwidth_matrix=None, n_components=None):
    """
    Custom KDE with optional PCA-based dimensionality reduction and a specified covariance matrix.
    """
    n_samples, n_dim = data.shape
    if n_components and n_components < n_dim:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
        print(f"Reduced data shape: {data.shape}")
        if bandwidth_matrix is None:
            bandwidth_matrix = np.cov(data, rowvar=False)

    if bandwidth_matrix is None:
        bandwidth_matrix = np.cov(data, rowvar=False)
    
    if not np.all(np.linalg.eigvals(bandwidth_matrix) > 0):
        raise ValueError("Bandwidth matrix must be positive definite.")

    n_samples, n_dim = data.shape
    log_const = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(np.linalg.det(bandwidth_matrix)))
    inv_bandwidth = np.linalg.inv(bandwidth_matrix)

    kde_values = []
    for i in range(n_samples):
        diff = data - data[i]
        log_probs = log_const - 0.5 * np.einsum('ij,jk,ik->i', diff, inv_bandwidth, diff)
        kde_value = np.sum(np.exp(log_probs))
        kde_values.append(kde_value / n_samples)

    return np.array(kde_values)


def filter_data_by_kde(combined_list, lower_threshold, upper_threshold, kde_file):
    """
    Filters data points based on KDE values, saving/loading KDE values to/from a file.
    """
    data = np.array(combined_list).T

    # Check if KDE file exists
    if os.path.exists(kde_file):
        print(f"Loading KDE values from {kde_file}...")
        with open(kde_file, 'rb') as f:
            kde_values = pickle.load(f)
    else:
        print("Computing KDE values...")
        distance_matrix_ = distance_matrix(data.T, data.T)
        upper_triangle_values = distance_matrix_[np.triu_indices_from(distance_matrix_, k=1)]
        upper_triangle_values.sort()

        one_percent = int(0.01 * len(upper_triangle_values))
        bandwidth = upper_triangle_values[one_percent]
        print("Bandwidth:", bandwidth)

        bandwidth_matrix = bandwidth * np.identity(data.shape[0])
        kde_values = custom_kde(data.T, bandwidth_matrix)

        # Save KDE values to file
        with open(kde_file, 'wb') as f:
            pickle.dump(kde_values, f)

    print("KDE values:", kde_values)

    # Filter data based on KDE thresholds
    filtered_data = [point for point, kde_val in zip(data.T, kde_values) if lower_threshold <= kde_val <= upper_threshold]
    removed_data = [point for point, kde_val in zip(data.T, kde_values) if kde_val < lower_threshold or kde_val > upper_threshold]
    
    return np.array(filtered_data), np.array(removed_data), kde_values


def load_embedding(ds_name, embeddings_dir='data/embeddings'):
    """Load embeddings only once to avoid redundant loading."""
    embedding_path = None
    for _embedding in os.listdir(embeddings_dir):
        if _embedding.startswith(ds_name):
            embedding_path = os.path.join(embeddings_dir, _embedding)
            break

    if embedding_path is None:
        raise FileNotFoundError(f"No embedding file found starting with '{ds_name}' in {embeddings_dir}")

    with open(embedding_path, 'rb') as f:
        embedding_dct = pickle.load(f)
    
    return embedding_dct


def compute_energy_distance_matrix(ds_name, lower_kde_threshold, upper_kde_threshold, kde_file):
    """Compute the energy distance matrix for the embedding file that starts with 'ds_name'."""
    embedding_dct = load_embedding(ds_name)
    ids = list(embedding_dct.keys())
    combined_list = []

    # Combine all embedding arrays into a single list for KDE filtering
    for id_ in ids:
        for embedding_array in embedding_dct[id_]:
            combined_list.extend(embedding_array.reshape(-1, embedding_array.shape[-1]))

    # Filter data using KDE with the specified thresholds
    filtered_data, removed_data, kde_values = filter_data_by_kde(combined_list, lower_kde_threshold, upper_kde_threshold, kde_file)

    # Create a deep copy of the original embedding dictionary for modification
    temp_dict = deepcopy(embedding_dct)

    # Convert filtered_data to a list of lists for easier comparison
    filtered_list_final = filtered_data.tolist()

    # Filter the dictionary by removing embeddings identified as outliers
    for key, value in embedding_dct.items():
        filtered_values = [v for v in value if v.tolist() in filtered_list_final]
        temp_dict[key] = np.array(filtered_values)

    # Initialize energy distance matrix
    energy_distance_matrix = np.zeros(shape=(len(ids), len(ids)))

    # Compute energy distances
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_i = ids[i]
            id_j = ids[j]
            embedding_i = temp_dict[id_i]
            embedding_j = temp_dict[id_j]
            
            if embedding_i.size > 0 and embedding_j.size > 0:
                energy_distance = dcor.energy_distance(embedding_i, embedding_j)
                energy_distance_matrix[i, j] = energy_distance
                energy_distance_matrix[j, i] = energy_distance
            else:
                print(f"Warning: One or both of the embeddings for {id_i} and {id_j} are empty after filtering.")

    return energy_distance_matrix
