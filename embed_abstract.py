"""
Yuli Tshuva
"""

import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings
from transformers import BertModel, BertTokenizer
import torch
warnings.filterwarnings("ignore")

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def make_embedding_file(name, source_path):
    """
    Create an embedding file for a given dataset.
    :param name:  Name of the dataset
    :param source_path:  Path to the dataset (the vertices.csv file)
    :return:
    """
    # Read the data
    data = pd.read_csv(source_path)
    print(f"Creating embeddings for {name} dataset from '{source_path}'...")
    # Get abstracts
    abstracts, ids = list(data['abstract'].fillna("")), list(data['id'])
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
    print(f"Saving embeddings to 'data/embeddings/{name}_embeddings.pkl'...")
    with open(f"data/embeddings/{name}_embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict  # Return the embeddings


def make_bert_embedding(name, source_path):
    """
        Create an embedding file for a given dataset.
        :param name:  Name of the dataset
        :param source_path:  Path to the dataset (the vertices.csv file)
        :return:
        """
    # Load BERT model and tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Read the data
    data = pd.read_csv(source_path)
    print(f"Creating embeddings for {name} dataset from '{source_path}'...")
    # Get abstracts
    abstracts, ids = list(data['abstract'].fillna("")), list(data['id'])
    # Split into sentences
    abstracts = [abstract.split('. ') for abstract in abstracts]
    # Set a dictionary to save the embeddings
    embeddings_dict = {}
    # Iterate through the abstracts and encode them
    for i, (_id, abstract) in enumerate(zip(ids, abstracts)):
        # Add back the '.' to the end of each sentence
        abstract = [sentence + '.' for sentence in abstract]
        for a in abstract:  # for each sentence.
            inputs = tokenizer.encode(a, return_tensors='pt')
            # make sure the length is 512.
            if len(inputs[0]) > 512:
                inputs = inputs[:, :512]
            with torch.no_grad():
                outputs = model(inputs)
                curr_output = torch.Tensor(outputs[0].mean(dim=1).numpy()[0])
                if _id in embeddings_dict:
                    embeddings_dict[_id].append(curr_output)
                else:
                    embeddings_dict[_id] = [curr_output]

        if i % 50 == 0:
            print(f"Finished encoding {i} abstracts.")
    # Save the embeddings
    print(f"Saving embeddings to 'data/embeddings/{name}_embeddings.pkl'...")
    with open(f"data/embeddings_bert/{name}_embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict  # Return the embeddings
