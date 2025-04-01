import os
import pickle
import numpy as np
import dcor
from typing import List
from copy import deepcopy
from scipy.stats import gaussian_kde
from embed_abstract import make_embedding_file  # Change to relative import in pip version
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree


def load_embedding(ds_name, embeddings_dir='data/embeddings'):
    """Load embeddings only once to avoid redundant loading."""
    embedding_path = None
    for _embedding in os.listdir(embeddings_dir):
        if _embedding.startswith(ds_name):
            embedding_path = os.path.join(embeddings_dir, _embedding)
            break

    if embedding_path is None or not os.path.exists(embedding_path):  # If the embedding file does not exist, create it.
        path = input(f"Embedding file for {ds_name} does not exist. Enter the path to the dataset: ")
        embedding_dct = make_embedding_file(ds_name, path)
        return embedding_dct

    with open(embedding_path, 'rb') as f:
        embedding_dct = pickle.load(f)

    return embedding_dct


def load_kde_vals(ds_name, kde_dir='data/kde_values'):
    """Load KDE values only once to avoid redundant loading."""
    kde_path = None
    for _kde in os.listdir(kde_dir):
        if _kde.startswith(ds_name):
            kde_path = os.path.join(kde_dir, _kde)
            break

    if kde_path is None:
        return None

    with open(kde_path, 'rb') as f:
        kde_vals = pickle.load(f)

    return kde_vals


def get_kde_values(ds_name, embeddings_dict, n_components_=5):
    """
    Reduce dimensionality using PCA and compute KDE values.
    Save KDE values to a file.
    :param ds_name:  Name of the dataset
    :param embeddings_dict:  Dictionary of embeddings
    :param n_components_:   Number of components for PCA
    :return:
    """
    values = []  # Combine all embedding arrays into a single list for KDE filtering
    id_map = {k: [] for k in embeddings_dict.keys()}  # Map ID to sentence embeddings

    for k, v in embeddings_dict.items():
        for i in range(v.shape[0]):
            values.append(v[i])
            id_map[k].append((v[i], 0))  # Map sentence embedding to its ID

    values = np.array(values)  # Convert to numpy array

    vals = PCA(n_components=n_components_).fit_transform(values)
    values = vals.T

    kde = gaussian_kde(values)

    # Compute KDE values
    for k in id_map.keys():
        for i in range(len(id_map[k])):
            id_map[k][i] = (id_map[k][i][0], kde.logpdf(vals[i]))  # Update KDE values

    # Save KDE values to a file
    kde_file = f'data/kde_values/{ds_name}.pkl'
    try:
        with open(kde_file, 'wb') as f:
            pickle.dump(id_map, f)
    except OSError:
        dir_ = 'data/kde_values'
        os.makedirs(dir_, exist_ok=True)  # Create directory if it does not exist
        with open(kde_file, 'wb') as f:
            pickle.dump(id_map, f)

    return id_map


def filter_data(ds_name, embedding_dict, _lb, _ub, n_components_=5):
    """
    Filter data points based on KDE values.
    :param ds_name:  Data to be filtered
    :param embedding_dict:  Dictionary of embeddings
    :param _lb:  Lower threshold for KDE values
    :param _ub:  Upper threshold for KDE values
    :param n_components_:  Number of components for PCA
    """
    kde_vals = load_kde_vals(ds_name)  # Load KDE values
    if kde_vals is None:
        kde_vals = get_kde_values(ds_name, embedding_dict, n_components_)  # Compute KDE values.

    for k, values in kde_vals.items():  # Filter embeddings based on KDE values
        embeddings_to_keep = []
        for v in values:
            if _lb <= v[1] <= _ub:
                embeddings_to_keep.append(v[0])
        embedding_dict[k] = np.array(embeddings_to_keep)  # Update embeddings

    return embedding_dict  # Return filtered embeddings


def compute_energy_distance_matrix(ds_name, _lb, _ub, n_components_=5):
    """Compute the energy distance matrix for the embedding file that starts with 'ds_name'."""
    embedding_dict = load_embedding(ds_name)  # Load embeddings

    # Filter embeddings based on KDE values
    filtered_data = filter_data(ds_name, embedding_dict, _lb, _ub, n_components_)
    ids = list(filtered_data.keys())

    # Create a deep copy of the original embedding dictionary for modification
    temp_dict = deepcopy(filtered_data)

    # Filter the dictionary by removing embeddings identified as outliers
    for key, value in filtered_data.items():
        temp_dict[key] = np.array(value)  # Convert list to numpy array

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

    # Save the energy distance matrix to a file
    filename = 'data/distances/' + ds_name + '_energy_distance_matrix.pkl'
    try:
        with open(filename, 'wb') as f:
            pickle.dump(energy_distance_matrix, f)
    except OSError:  # Create directory if it does not exist
        dir_ = 'data/distances'
        os.makedirs(dir_, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(energy_distance_matrix, f)

    return energy_distance_matrix


def min_max_normalize_non_diagonal(matrix):
    # Ensure the matrix is a NumPy array
    matrix = np.array(matrix, dtype=np.float64)

    # Get the non-diagonal elements
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    non_diag_elements = matrix[mask]

    # Get actual min and max values of the non-diagonal elements
    min_val = non_diag_elements.min()
    max_val = non_diag_elements.max()

    # Avoid division by zero in case all values are the same
    if max_val > min_val:
        scaled = (non_diag_elements - min_val) / (max_val - min_val) * (1 - 1e-8) + 1e-8
    else:
        scaled = np.full_like(non_diag_elements, (1e-8 + 1) / 2)

    # Assign the scaled values back to the original matrix
    matrix[mask] = scaled

    return matrix


def compute_probability_matrix(embedding_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Computes p(i, j) where each entry counts the number of times a vector in set i has its
    closest neighbor in set j, ensuring that a vector is never matched to itself.

    Args:
        embedding_arrays (List[np.ndarray]): List of NumPy arrays, each containing embeddings as rows.

    Returns:
        np.ndarray: A probability matrix p(i, j) of shape (num_sets, num_sets).
    """
    num_sets = len(embedding_arrays)
    count_matrix = np.zeros((num_sets, num_sets))  # Initialize count matrix

    for i in range(num_sets):
        # Exclude embeddings from set i
        other_sets = [embedding_arrays[j] for j in range(num_sets) if j != i]
        all_vectors = np.vstack(other_sets)  # Merge remaining sets
        set_indices = np.concatenate([[j] * len(embedding_arrays[j]) for j in range(num_sets) if j != i])

        # Build BallTree for other sets
        tree = BallTree(all_vectors, metric="euclidean")

        if i % 20 == 0:
            print(f"Processing set {i}/{num_sets}...")

        # Query the tree for each vector in set i
        for v in embedding_arrays[i]:
            _, closest_idx = tree.query([v], k=1)  # Find the closest neighbor (since i's vectors are excluded)
            closest_set = set_indices[closest_idx[0][0]]
            count_matrix[i, closest_set] += 1  # Increment count

    # make the count matrix symmetric
    count_matrix = (count_matrix + count_matrix.T) / 2

    column_sums = count_matrix.sum(axis=0, keepdims=True)  # Sum of each column

    count_matrix = count_matrix / column_sums  # Normalize each column

    return count_matrix


def probability_to_distance(prob_matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Converts a probability matrix to a distance matrix using the formula:
    distance(i, j) = exp(-alpha * probability(i, j))

    Args:
        prob_matrix (np.ndarray): A probability matrix of shape (num_sets, num_sets).
        alpha (float): A scaling factor for the distances.

    Returns:
        np.ndarray: A distance matrix of shape (num_sets, num_sets).
    """
    return np.exp(-alpha * prob_matrix)


def make_distance_matrix(ds_name: str, alpha: float = 1.0) -> np.ndarray:
    """
    Computes a distance matrix from a list of embedding arrays.

    Args:
        ds_name (str): Name of the dataset.
        alpha (float): A scaling factor for the distances.

    Returns:
        np.ndarray: A distance matrix of shape (num_sets, num_sets).
    """

    embedding_dict = load_embedding(ds_name)

    embedding_arrays = [embedding_dict[key] for key in embedding_dict]
    prob_matrix = compute_probability_matrix(embedding_arrays)
    dist_matrix = probability_to_distance(prob_matrix, alpha)

    # normalize.
    dist_matrix = min_max_normalize_non_diagonal(dist_matrix)

    # save the distance matrix to a file
    filename = 'data/distances_app/' + ds_name + '_distance_matrix.pkl'
    os.makedirs('data/distances_app/', exist_ok=True)  # Create directory if it does not exist
    with open(filename, 'wb') as f:
        pickle.dump(dist_matrix, f)

    return dist_matrix


"""DS_NAMES = ['apple', 'car', 'clock', 'London', 'turtle', '3D printing', 'additive manufacturing', 'autonomous drones',
            'composite material', 'hypersonic missile', 'nuclear reactor', 'quantum computing', 'scramjet',
            'smart material', 'wind tunnel']

for ds_name in DS_NAMES:
    print(f"Computing energy distance matrix for {ds_name}...")
    compute_energy_distance_matrix(ds_name, 0.1, 5.0, 5)"""
