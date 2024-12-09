"""
Yuli Tshuva
"""

import os
from os.path import join
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = 'data/wikipedia'
EMBEDDINGS_PATH = 'data/embeddings'

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# Iterate over all files in the data folder
for file_name in os.listdir(DATA_PATH):
    print(file_name)
    # Skip the graph files
    if "edges" in file_name:
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
    for _id, abstract in zip(ids, abstracts):
        # Add back the '.' to the end of each sentence
        abstract = [sentence + '.' for sentence in abstract]
        # Encode the abstract
        embeddings = model.encode(abstract)
        # Save the embeddings
        embeddings_dict[_id] = embeddings
    # Save the embeddings
    name = file_name.split(".")[0]
    print(file_name.split(".")[0])
    with open(f"{EMBEDDINGS_PATH}/{name}_embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings_dict, f)
