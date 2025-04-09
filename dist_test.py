import numpy as np
from sklearn.neighbors import BallTree, KernelDensity
from scipy.spatial.distance import cdist
from typing import List, Dict
import pickle as pkl
import pandas as pd
from one_to_rule_them_all import compute_silhouette_score


def compute_probability_matrix(embedding_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Computes p(i, j) where each entry counts the number of times a vector in set i has its
    closest neighbor in set j, ensuring that a vector is never matched to itself.

    Args:
        embedding_arrays (List[np.ndarray]): List of NumPy arrays, each containing embeddings as rows.

    Returns:
        np.ndarray: A probability matrix p(i, j) of shape (num_sets, num_sets).
    """
    # num_sets = len(embedding_arrays)
    # count_matrix = np.zeros((num_sets, num_sets))  # Initialize count matrix

    # Flatten all embeddings into a single array and keep track of set indices
    # all_vectors = np.vstack(embedding_arrays)  # Merge all sets into one array
    # set_indices = np.concatenate([[j] * len(embedding_arrays[j]) for j in range(num_sets)])  # Track set membership

    # Step 1: Build a BallTree with all vectors
    # tree = BallTree(all_vectors, metric="euclidean")

    # original_set_copy = -1
    # original_set = 0

    # Step 2: Query the tree for each vector, finding the second closest neighbor
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


def make_distance_matrix(embedding_dict: Dict, alpha: float = 1.0) -> np.ndarray:
    """
    Computes a distance matrix from a list of embedding arrays.

    Args:
        embedding_arrays (List[np.ndarray]): List of NumPy arrays, each containing embeddings as rows.
        alpha (float): A scaling factor for the distances.

    Returns:
        np.ndarray: A distance matrix of shape (num_sets, num_sets).
    """
    embedding_arrays = [embedding_dict[key] for key in embedding_dict]
    prob_matrix = compute_probability_matrix(embedding_arrays)
    dist_matrix = probability_to_distance(prob_matrix, alpha)

    # normalize.
    dist_matrix = min_max_normalize_non_diagonal(dist_matrix)
    return probability_to_distance(prob_matrix, alpha)

# This part uses these functions:
def opposite_distances(data):
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                data[i][j] = 1 / data[i][j]
    return data


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


# test
emb_dict = pkl.load(open('data/embeddings/newsgroups_1k_sampled_embeddings.pkl', 'rb'))

print(type(emb_dict))
key = list(emb_dict.keys())[0]
print(key)
# print(emb_dict[key])

emb_list = [emb_dict[key] for key in emb_dict]
vertices = pd.read_csv('data/newsgroups_1k_sampled.csv')
dists = compute_probability_matrix(emb_list)

scaled_dists = min_max_normalize_non_diagonal(dists)
s = compute_silhouette_score('newsgroups_1k_sampled', vertices, scaled_dists, print_info=True)
s2 = compute_silhouette_score('newsgroups_1k_sampled', vertices, dists, print_info=True)
