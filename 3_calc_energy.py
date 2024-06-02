"""
Yuli Tshuva
Calculating the energy distance of the bags
"""

import dcor
import os
from os.path import join
import pickle
from tqdm.auto import tqdm
import numpy as np

# Embeddings directory
EMBEDDINGS_DIR = 'embeddings'
DISTANCES_DIR = 'distances'

# Iterate through the embeddings and calculate the energy distance
for embedding in tqdm(os.listdir(EMBEDDINGS_DIR)):
    # Get full path
    embedding_path = join(EMBEDDINGS_DIR, embedding)
    # Load the embedding
    with open(embedding_path, 'rb') as f:
        embedding_dct = pickle.load(f)
    # Get the ids of papers
    ids = list(embedding_dct.keys())

    # Calculate the energy distance
    energy_distance_matrix = np.zeros(shape=(len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id_i = ids[i]
            id_j = ids[j]
            embedding_i = embedding_dct[id_i]
            embedding_j = embedding_dct[id_j]
            energy_distance = dcor.energy_distance(embedding_i, embedding_j)
            energy_distance_matrix[i, j] = energy_distance
            energy_distance_matrix[j, i] = energy_distance

    print(energy_distance_matrix)
    break

    # Save the energy distance matrix
    # distances_path = join(DISTANCES_DIR, embedding)
    # save_dict = {"Distances": energy_distance_matrix, "IDs": ids}
    # with open(distances_path, 'wb') as f:
    #     pickle.dump(save_dict, f)
