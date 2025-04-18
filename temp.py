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
from functions import evaluate_clusters
#
# # Sample 5k vertices from the newsgroups dataset.
# df = pd.read_csv('data/reuters.csv')
# sampled_df = df.sample(1000)  # Sample 1000 rows from the dataframe
#
# sampled_df['id'] = range(1000)
#
# sampled_df.to_csv('data/reuters_1k.csv', index=False)
name = "posts_1k_sampled"
metrics_dct = json.load(open('metrics.json', 'r'))
scores = {}
for k, v in metrics_dct.items():
    scores[k] = v.get(name, None)
vertices = pd.read_csv(f'data/{name}.csv')

plot_bar(name, scores, vertices)
plot(name, vertices)
