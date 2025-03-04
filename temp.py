import pandas
import pickle as pkl
import networkx as nx
import numpy as np
import functions
import pandas as pd
from one_to_rule_them_all import (compute_purity_score, compute_vmeasure_score,
                                  compute_silhouette_score, compute_jaccard_index)

name = 'newsgroups_1k_sampled'
G = functions.load_graph('newsgroups_1k_sampled')

vertices = pd.read_csv('data/newsgroups_1k_sampled.csv')

dists = pkl.load(open('data/distances/newsgroups_1k_sampled_energy_distance_matrix.pkl', 'rb'))

print(compute_vmeasure_score(name, vertices, print_info=True))

print(compute_purity_score(name, vertices, print_info=True))

print(compute_silhouette_score(name, vertices, dists, print_info=True))

print(compute_jaccard_index(name, vertices, print_info=True))

# print the amount of clusters.
clusters = set(G.nodes()[v]['color'] for v in G.nodes)

print(f"Number of clusters: {len(clusters)}")