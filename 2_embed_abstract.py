"""
Yuli Tshuva
"""

import os
from os.path import join
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = 'data/graphs'
EMBEDDINGS_PATH = 'embeddings'

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# Iterate over all files in the data folder
for file_name in os.listdir(DATA_PATH):
    # Skip the graph files
    if "graph" in file_name:
        continue
    # Get full path
    file_path = join(DATA_PATH, file_name)
    # Read the data
    df = pd.read_csv(file_path)
    # Get abstracts
    abstracts, ids = list(df['abstract'].fillna("")), list(df['id'])
    # Split into sentences
    abstracts = [abstract.split('. ') for abstract in abstracts]
    # Set a dictionary to save the embeddings
    embeddings_dict = {}
    # Iterate through the abstracts and encode them
    for id, abstract in zip(ids, abstracts):
        # Add back the '.' to the end of each sentence
        abstract = [sentence + '.' for sentence in abstract]
        # Encode the abstract
        embeddings = model.encode(abstract)
        # Save the embeddings
        embeddings_dict[id] = embeddings
    # Save the embeddings
    with open(join(EMBEDDINGS_PATH, f'{file_name.split(".")[0]}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings_dict, f)
