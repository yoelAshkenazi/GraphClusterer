# Graph Clustering and Summarizing
We aim to improve the quality of a multi-document summary, given the connections between the texts (via shared authors, institution, subject etc.). 
Our method is as follows:
1. Calculate the direct distances between the texts in the graph. 
2. Build a new graph with two edge types: 
    * Blue edges: represent the original connections given.
    * Red edges: represent similarity between texts.
3. Cluster the multi color edged graph.
4. Summarize each cluster.
5. Evaluate the summarization of each cluster.

Our python package contains a full pipeline that performs the above operations (1-5), or alternatively perform specific operations individually.


## Table of Contents

-  [Installation](#installation)
-  [Quick tour](#quick-tour)
-  [Examples](#examples)

## Installation
We do not have an official python package manager installation available yet. However, 
this repository is public, and one can clone the repository and open it in any IDE that accepts git.


## Quick tour
To immediately use our package, you only need to run three functions (two if you wish to skip the final evaluations).<br>
You need to have your data in one of the following formats:
  1. a prepared graph of texts in '.gpickle' format.
  2. two csv files to make the graph from: vertex metadata and edges.

### Case 1
In this case you can skip the graph processing part, and go straight to the [clustering](#clustering) part.
The prepared graph needs to include the followings:
  1. vertex attribute called 'content', whose value is the textual data of the vertex.
  2. (optional) vertex attribute called 'shape', in order to distinguish between texts in different languages (for now, this case deals with each language individually).

### Case 2
In this case, we need to create the graph first.
1. The csv containing vertex data has to be named '{dataset name}_papers.csv', and has to contain at least an ID column and 'abstract' column.
The ID column values are used to identify texts, and the 'abstract' column should contain the texts.<br>
2. The csv containing edges has to be named '{dataset name}_graph.csv' and have 2 columns by default, and each of its rows has to contain at least one identifiable text that appears in the vertex data file.
In case the other element is not identifiable, a new vertex will be created, with a different type and no content.
3. Rows with 3 or more elements will be dealt as a clique.

Secondly, calculation of distances between texts are required. For that we firstly have to convert the texts to sets of embedding vectors.
We do that in '2_embed_abstracts.py' (___REGEV CONTINUE HERE___), and then we refer to each text as a set of sentence embeddings, and calculate 'energy distance' between 
two texts (two sets of vectors) in 'calc_energy.py'.
In order to successfully estimate the distance between two texts, we filter sentences that interrupt the procedure (in our case, we prefer to filter out the most common sentences, as well as the rarest sentences). We do that by combining all of the sentences' embeddings in the dataset into one list, order the items according to their frequencies, and filtering `least_cutoff_percentage` embeddings from the least common sentences. Similarly, we filter `most_cutoff_percentage` embeddings from the most common sentences. Both cutoff parameters are optimized for each dataset given.

After filtering out the irrelevant embeddings, we compute the distances between each pair of texts using 'energy distance':
```python
def compute_energy_distance_matrix(ds_name, least_cutoff_percentage, most_cutoff_percentage):
```

Where:
- `ds_name`: The name of the dataset (mandatory).
- `least_cutoff_percentage`: The percentage of data to filter among rare sentences.
- `most_cutoff_percentage`: The percentage of data to filter among frequent sentences.
We then calculate the energy distance between each pair of embeddings using the using `dcor` package and then symmetrically assign the distance to the matrix.


Returns: The energy distance matrix for the matched embedding.

You can then run the graph creating part of our pipeline in 'functions.py':
```python
def make_graph(name, **kwargs):
```
Where:
- `name`: The name of the dataset (the same name as in the file names). This value is mandatory, and the code will return an error without it.
- `**kwargs`: A dictionary of values to manually configure the graph creation. Here is an example dictionary:
```python
graph_kwargs = {
        'size': 2000,
        'color': '#1f78b4',
        'K': 5,
    }
```
where:
  * `k`: The number of neighbors to account for in the distance based edges.
  * `color`: The default vertex color (needed in order to plot the final vertex partition).
  * `size`: The default vertex size (needed in order to plot the final vertex partition).


Returns: Processed graph (networkx.Graph object)

### Clustering
After processing/creating the graph of texts (and keywords), we cluster the graph using the [Louvain method](https://en.wikipedia.org/wiki/Louvain_method) with the implementation embedded in the `networkx` module.
In case the graph consists of keyword vertices, and not exclusively texts, the other vertices are included in the clustering part of our pipeline, but ignored from then on.
The clustering is also done in the `function.py` source file. In addition to return the partition, our method also assigns each vertex with a color, as a way to map the vertices to clusters later on.
```python
def cluster_graph(G, name, **kwargs):
```
Where:
* `G`: The processed/given graph.
* `name`: The name of the dataset.
* `**kwargs`: A dictionary of values to manually configure the clustering process. Here is an example dictionary:
```python
clustering_kwargs = {
        'save': True,
        'resolution': res,
        'weight': weight,
        'K': K
    }
```
Where:
* `save`: A flag indicating whether to save the clustered graph as a `.gpickle` format or not.
* `weight`: The assigned weight for distance based edges.
* `k`: The number of neighbors to account for in the distance based edges.


Returns: $\mathcal{P}$ a partition of vertices from `G` into communities.

### Summarization
(***REGEV CONTINUE HERE***)

### Evaluation
(***REGEV CONTINUE HERE***)
