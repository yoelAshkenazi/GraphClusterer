import pandas
import pickle as pkl
import networkx as nx
import numpy as np
import functions
import pandas as pd
import json
from one_to_rule_them_all import (compute_purity_score, compute_vmeasure_score,
                                  compute_silhouette_score, compute_jaccard_index, plot_bar)
from plot import plot

# Sample 5k vertices from the newsgroups dataset.
# df = pd.read_csv('data/Terrorism_1000_samples_nodes.csv')
# # sampled_df = df.sample(5000)
# sampled_df = df
# # change the 'content' column to 'abstract'. (with the same texts)
# sampled_df = sampled_df.rename(columns={'content': 'abstract', 'id': 'page name'})
# sampled_df['id'] = range(1000)
#
# sampled_df.to_csv('data/terrorism_wiki.csv', index=False)

metrics_dict = json.load(open("metrics.json", 'r'))

scores = {}
for k, v in metrics_dict.items():
    scores[k] = v['terrorism_wiki']

plot_bar('Terrorism_wiki', scores)