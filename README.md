# Graph Clustering and Summarizing
## Introduction
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
-  [Running example](#running-example)
-  [Quick tour](#quick-tour)

## Installation
In a clean environment, simply type
```python
pip install GraphClusterer
```
to download our python package, or alternatively clone this repository if you wish to manually tune the pipeline (recommended).
There is one difference between the usage of the package compared to the clone repository, which is a change in the `.env` file that is required for the repository, compared to the inclusion of the API keys needed as strings in the `config.json` file when using the package (see example in [Quick tour](#quick-tour)).

## Running example
There are two ways to run and use our project.
1. using the python package (GraphClusterer)
2. using a cloned version of this repository
```input
|-- vertices.csv
|-- edges.csv
|-- config.json
|-- distance_matrix.pkl (optional)
|-- .env (required for the second case)
`-- main.py
```
And after running the code, the following directories will be added:
```output
|-- data
|   `-- clusteder_graphs
|       `-- name.gpickle
|
|-- Results
|   |-- Summaries
|   |   `-- name
|   |       |-- title_1.txt
|   |       |-- ...
|   |       `-- title_n.txt
|   |
|   |-- starry
|   |   |-- name.csv
|   |   `-- name_titles.csv
|   |
|   |-- plots
|   |   `-- name.png
|   |
|   `-- html
|       `-- name.html
|
`-- metrics.json
```
If `distance_matrix` is provided and valid (see [file formats](#file-formats) below), the following directories will be added to `data` after running:
```
data
|-- embeddings
|   `-- vertices_file_name_embeddings.pkl
|
`-- kde_values
    `-- name.pkl
```
Which cache the embedding vectors for each sentence, and the KDE values for filtering (**NEED TO FULLY INTEGRATE MANUAL FILTERING INTO THE PIPELINE**).

### File formats

#### main.py
Here's an example script for the main file in the first case usage (the python package). If you intend on using the cloned repository, you can use the `main.py` file [here](https://github.com/yoelAshkenazi/GraphClusterer/blob/master/main.py).
```python

from GraphClusterer.main import run_pipeline

if __name__ == '__main__':

    run_pipeline()
```

You can execute the function with no arguments firstly, in order to have the directory tree needed, and also a default dataset. However, We also allow a generic LLMProvider class to be a part of the input in later executions, thus allowing the usage of other LLMs in the pipeline. The requested class must include the following methods:
```python

class LLMProvider:
   def __init__(self, **kwargs):...  # build method.
   def generage_response(self, prompt, max_tokens):...  # generate a response of at most max_tokens length to the given prompt.
   def reconnect(self):...  # if using an API service, reconnect to the service to avoid overloads. If using a local model, this function can return self.
```

And respectively, the `run_pipeline` method can accept the following:
```python
run_pipeline(params: Dict, summary_model: LLMProvider, refinement_model: LLMProvider)
```

If `summary_model` is not provided, the pipeline will use the [`Cohere`](https://docs.cohere.com/v2/reference/about) LLM be default, and if `refinement_model` is not provided, the `Llama` LLM will be used.

### config.json
The `config.json` file requires core elements, and additional keys when using the python package, in order to avoid the usage of a `.env` file.
```json
{
  "graph_kwargs": {
    "size": 2000,
    "K": 5,
    "color": "#1f78b4"
  },

  "clustering_kwargs": {
    "method": "louvain",
    "resolution": 0.5,
    "save": true
  },

  "draw_kwargs": {
    "save": true,
    "method": "louvain",
    "shown_percentage": 0.3
  },

  "name": "clock",
  "vertices_path": "clock_nodes.csv",
  "edges_path": "clock_edges.csv",
  "distance_matrix_path": "optional path to distance_matrix.pkl",

  "iteration_num": 1,
  "print_info": true,
  "update_factor": 0.5,
  "cohere_key": "your Cohere API key here",
  "llama_key": "your Llama API key here"
}
```

### .env
When using the cloned repository, you must have a `.env` file in the same working directory as `main.py`.
```.env
COHERE_API_KEY="your Cohere API key here"
REPLICATE_API_TOKEN="your Llama API key here"
```

### vertices.csv
The pipeline expects a vertices file with the following structure for each line (text can be empty).
`id` | `abstract` (includes the text to summarize) | `additional attributes` (e.g. color, language, shape etc. Used for debugs and plotting. Each attribute in its own column)
:---: | :---: | :---:
vertex_1_id | vertex_1_abstract_text | vertex_1_attributes

### edges.csv
The pipeline expects a file containing the original edges (for distance-based edges see [here](#)). Each edge (row in the file) should have either 2 or 3 columns (if 3, all rows need to have 3 columns).
- The 2 column case occurs when only textual vertices are introduced in the graph.
- The 3 column case occurs when there are additional vertices without text in the edges file. In that case, the new vertices are added to the graph with their specified type (in the third column).

### distance_matrix.pkl
The distance matrix needs to be an array of size `NxN`, where N is the number of textual vertices (rows in `vertices.csv`). If provided, the distance matrix is used to add a second type of edges ([red edges](#introduction)).


## Quick tour
To immediately use our package, you only need to use two functions.<br>

In order to use the package (or cloned repository) you need to prepare a configuration file in advanced (see [`config.json`](https://github.com/yoelAshkenazi/GraphClusterer/blob/master/config.json)).
There are two use cases: 
1. cloned repository: you also need to have a `.env` file in the same directory as `main.py`, in which the API keys for llama and cohere are kept like the example here:
  ```.env
COHERE_API_KEY=[your API key for cohere (string format)]
REPLICATE_API_TOKEN=[your API token for llama (string format)]
```
2. python package: you also need to include the following arguments in your `config.json` file:
   ```config.json
   'cohere_key': [your API key for cohere (string format)],
   'llama_key': [your API key for llama (string format)]
   ```

In both cases, your `main.py` code should be like:
```python
params = load_params(config_path)
# Set the parameters for the pipeline.
pipeline_kwargs = {
   'graph_kwargs': params['graph_kwargs'],
   'clustering_kwargs': params['clustering_kwargs'],
   'draw_kwargs': params['draw_kwargs'],
   'print_info': params['print_info'],
   'iteration_num': params['iteration_num'],
   'vertices': pd.read_csv(params['vertices_path']),
   'edges': pd.read_csv(params['edges_path']),
   'distance_matrix': get_distance_matrix(params['distance_matrix_path']),
   'name': params['name'],
    }

# Run the pipeline.
one_to_rule_them_all.the_almighty_function(pipeline_kwargs)
```

Alternatively, you can manually perform each sub-task in our pipeline using the following functions:
1. Create the graph with two edge colors: `functions.make_graph(**graph_kwargs)`
2. Cluster the graph: `functions.cluster_graph(**clustering_kwargs)`
3. Summarize each cluster: `summarize.summarize_per_color(**kwargs)` (need to divide the graph into clusters and input a list of subgraphs)
4. Evaluate the clusters and summaries: several functions in the `evaluate` module (see [here](https://github.com/yoelAshkenazi/GraphClusterer/blob/master/evaluate.py))

You can also use the old versions and follow these instructions:
### Case 1
In this case you can skip the graph processing part, and go straight to the [clustering](#clustering) part.
The prepared graph needs to include the followings:
  1. vertex attribute called 'content', whose value is the textual data of the vertex.
  2. (optional) vertex attribute called 'shape', in order to distinguish between texts in different languages (for now, this case deals with each language individually).

### Case 2
In this case, we need to create the graph first.
1. The csv containing vertex data has to be named '`{dataset name}_papers.csv`', and has to contain at least an ID column and 'abstract' column.
The ID column values are used to identify texts, and the 'abstract' column should contain the texts.<br>
2. The csv containing edges has to be named '`{dataset name}_graph.csv`' and have 2 columns by default, and each of its rows has to contain at least one identifiable text that appears in the vertex data file.
In case the other element is not identifiable, a new vertex will be created, with a different type and no content.
3. Rows with 3 or more elements will be dealt as a clique.

Secondly, calculation of distances between texts are required. For that we firstly have to convert the texts to sets of embedding vectors.
We do that in '`2_embed_abstracts.py`' using the `sentence-transformers` package, refer to each text as a set of sentence embeddings, and calculate 'energy distance' between two texts (two sets of vectors) in '`calc_energy.py`'.
In order to successfully estimate the distance between two texts, we filter sentences that interrupt the procedure (in our case, we prefer to filter out the most common sentences, as well as the rarest sentences). We do that by combining all of the sentences' embeddings in the dataset into one list, order the items according to their frequencies, and filtering `least_cutoff_percentage` embeddings from the least common sentences. Similarly, we filter `most_cutoff_percentage` embeddings from the most common sentences. Both cutoff parameters are optimized for each dataset given.

After filtering out the irrelevant embeddings, we compute the distances between each pair of texts using 'energy distance':
```python
def compute_energy_distance_matrix(ds_name, least_cutoff_percentage, most_cutoff_percentage):
```

Where:
- `ds_name`: The name of the dataset.
- `least_cutoff_percentage`: The percentage of data to filter among rare sentences.
- `most_cutoff_percentage`: The percentage of data to filter among frequent sentences.
We then calculate the energy distance between each pair of embeddings using the using `dcor` package.


Returns: The energy distance matrix for the matched embedding.


You can then run the graph creating part of our pipeline in '`functions.py`':
```python
def make_graph(name, **kwargs):
```
Where:
- `name`: The name of the dataset.
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


Returns: Processed graph (`networkx.Graph` object)
   
<img src="https://github.com/user-attachments/assets/e2715659-458f-41e5-b16b-7879c5aef7e0" alt="0" width="600px">

### Clustering
After processing/creating the graph of texts (and keywords), we cluster the graph using the [Louvain method](https://en.wikipedia.org/wiki/Louvain_method) with the implementation embedded in the `networkx` module.
In case the graph consists of keyword vertices, and not exclusively texts, the other vertices are included in the clustering part of our pipeline, but ignored from then on.
The clustering is also done in the '`function.py`' source file. In addition to return the partition, our method also assigns each vertex with a color, as a way to map the vertices to clusters later on.
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
* `save`: A flag indicating whether to save the clustered graph as a '`.gpickle`' format or not.
* `weight`: The assigned weight for distance based edges.
* `k`: The number of neighbors to account for in the distance based edges.


Returns: $\mathcal{P}$ a partition of vertices from `G` into communities.

Example for the clustering:

<img src="https://github.com/user-attachments/assets/54c32d5a-f4f5-4dc3-a144-ec990f7bb0b3" alt="unnamed" width="600px">


### Summarization
As mentioned in the [introduction](#introduction) section, the main goal of our pipeline is to summarize clusters of vertices, rather than the full graph.
To achieve that we perform the following steps:
1. Cluster the graph (see [clustering](#clustering)).
2. Divide the graph according to the clusters (done in `filter_by_color()`).
3. Summarize each cluster individually. In this step, each time we are given a subgraph of only the cluster. Firstly we        filter out non-textual vertices, then we access the texts and save them in a list, and then we send the list with a         prompt to two iterations of LLM generation:
      1. `command-r`: in this iteration, we generate a draft of the final summary.
      2. `llama-3.1`: in this iteration, we refine the summary and enrich its language.


   And a third iteration generates a title for the summary (also made with `llama-3.1`).


The full summarization process happen in `summarize_per_color()`:
```python
def summarize_per_color(subgraphs, name):
```

Where:
* `subgraphs`: A list of subgraphs, each belong to a different cluster (see step 2).
* `name`: The name of the dataset.


The function fetches the texts from the vertices in the subgraph, send them to the above 3 iterations of LLM generation, and saves the summaries for each cluster as a '`.txt`' file in a folder named `name`.

<img src="https://github.com/user-attachments/assets/3394ceaa-7660-4e79-803a-5d21836ec5b6" alt="0" width="600px">

## Example of a summary:
```text
Biomass-derived carbons (BDCs) and their composites with conductive materials, such as metals, metal sulfides, carbon nanotubes,
and reduced graphene oxide, are used to enhance the performance of supercapacitors. By combining BDCs with conductive additives,
researchers aim to improve conductivity, charge/discharge capabilities, and specific surface area, resulting in higher specific
capacitance values. This approach integrates the benefits of electrochemical double-layer capacitors (EDLCs) and pseudocapacitors,
leading to enhanced energy density. Layered double hydroxides (LDHs), synthetic two-dimensional nano-structured anionic clays, are
also explored as hosts for Azo-compounds to create nano-hybrid materials. Intercalating large anionic pigments like phenyl azobenzoic
sodium salt into Zn-Al LDH increases the interlayer spacing significantly, and the resulting nano-hybrid material is used as a filler
for polyvinyl alcohol (PVA) to form nano-composites that exhibit improved thermal stability compared to pure PVA.
```

### Evaluation
In the evaluation section, we execute a series of tests in order to assess the quality of:
* The [clustering](#clustering-scores).
* The [summaries](#summary-scores) (as text documents).
* The [consistency](#consistency-index) between a summary and its origin.


#### Clustering scores
The clustering scores evaluate how good the clustering partitioned the graph. For that we used two metrics (one is irrelevant if the graph is given by the user and not created by our '`make_graph()`' method). Here we computed everything by ourselves.
1. 'Average index': This metric measures the proportion between the average distance between two vertices from the same cluster, compared to two random vertices. (This method is relevant only for graphs our method created)
2. 'Largest cluster percentage': This metric measures the percentage of data in the largest cluster created.


Both metrics are between 0 and 1, and we would expect different optimal results:
* The optimal 'average index' should be as low as possible, but strictly positive.
* The optimal 'largest cluster percentage' shoule be around 0.5 (or 50% of the data).


#### Summary scores
The summary scores measure how understandable a summary is as a text. For that four scores are estimated:
1. 'Fluency'
2. 'Consistency'
3. 'Coherence'
4. 'Relevancy'


The estimation is made with an LLM judge (we used '`command-r`', but other models perform similarly here).
The scripts we used for each of the four metrics are from [this repository](https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-summarization) and [this tutorial](https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization).


#### Consistency index
In addition to the other metrics, we also estimated how much a given summary agrees with texts from its origin compared to texts from other clusters.

In order to estimate this metric, for each cluster we sampled texts from within and from outside it, then sent them with a specially designed prompt to an LLM judge (we used '`command-r`' here as well). 

<img src="https://github.com/user-attachments/assets/c1d93f94-d081-4a8d-97a2-919c8c810ad3" alt="0" width="600px">

### Results
At the end, there are two results files:
1. The Scores plot, which is saved both as a figure at `Results/plots/name.png`, and in a scores dictionary at `metrics.json` for further analysis if needed.

<img src="https://github.com/user-attachments/assets/5ab46dec-c4d3-4bbf-8959-ab813372b186" alt="3D printing" width="600px">

2. The interactive HTML graph, which is saved at `Results/html/name.html`

<img src="https://github.com/user-attachments/assets/29aad46e-6205-4e4f-9e1a-6740a22d32f7" alt="html preview" width="600px">
