import pickle as pkl
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import functions
G = functions.load_graph('3D printing')

abstracts = [n for n in G.nodes if G.nodes()[n]['type'] == 'paper']
G = G.subgraph(abstracts)

print(G.nodes(data=True))