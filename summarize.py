"""
Yoel Ashkenazi
this file is responsible for summarizing the result graphs.
"""
import os
import pandas as pd
from typing import List
import networkx as nx
import cohere
from dotenv import load_dotenv
import pickle as pkl
import replicate

def load_graph(name: str, version: str, proportion, k: int = 5, weight: float = 1, optimized: bool = False) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :param version: the version of the graph.
    :param k: the KNN parameter.
    :param proportion: the proportion of the graph.
    :param weight: the weight of the edges.
    :param optimized: whether to use the optimized version of the graph.
    :return:
    """

    if optimized:
        graph_path = None
        for file in os.listdir('data/optimized_graphs'):
            if file.startswith(name):
                graph_path = 'data/optimized_graphs/' + file
                break
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', " \
                                                                   "or 'proportion'."
        graph_path = f"data/processed_graphs/k_{k}/"
        if weight != 1:
            graph_path += f"weight_{weight}/"
        if version == 'distances':
            graph_path += f"only_distances/{name}.gpickle"

        elif version == 'original':
            graph_path += f"only_original/{name}.gpickle"

        else:
            if proportion != 0.5:
                graph_path += f"{name}proportion{proportion}.gpickle"
            else:
                graph_path += f"{name}.gpickle"

    # load the graph.
    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    return graph


def filter_by_colors(graph: nx.Graph) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to the community colors.
    :param graph: the graph.
    :return:
    """
    # first we filter articles by vertex type.
    articles = [node for node in graph.nodes() if graph.nodes.data()[node]['type'] == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    nodes = graph.nodes(data=True)
    colors = set([node[1]['color'] for node in nodes])
    subgraphs = []
    for i, color in enumerate(colors):  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]

        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)
        print(f"color {color} has {len(subgraph.nodes())} vertices.")

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph], name: str, version: str, proportion: float, save: bool = False,
                       k: int = 5, weight: float = 1, optimized: bool = False):
    """
    Summarizes each subgraph's abstract texts using Cohere's API, prints the results, and optionally saves them to text files.

    :param subgraphs: List of subgraphs.
    :param name: The name of the dataset.
    :param version: The version of the graph.
    :param proportion: The proportion of the graph.
    :param save: Whether to save the results.
    :param k: The KNN parameter.
    :param weight: The weight of the edges.
    :param optimized: Whether to use the optimized version of the graph.
    :return: None
    """

    # File path construction as per user-provided method
    if optimized:
        result_file_path = "Summaries/optimized/" + name + '/'
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', or 'proportion'."
        result_file_path = f"Summaries/k_{k}/weight_{weight}/{name}"
        if version == 'distances':
            result_file_path += '_only_distances/'
        elif version == 'original':
            result_file_path += '_only_original/'
        else:
            if proportion != 0.5:
                result_file_path += f'proportion{proportion}/'

    # Ensure the directory exists
    os.makedirs(result_file_path, exist_ok=True)

    # Clean previous summaries by removing existing files in the directory
    if os.path.exists(result_file_path):
        for file in os.listdir(result_file_path):
            os.remove(os.path.join(result_file_path, file))

    # Integrate dotenv to load environment variables
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)

    api_token = os.getenv("LLAMA_API_KEY")
    # Load the abstracts from the CSV file
    PATH = f'data/graphs/{name}_papers.csv'

    # Ensure the CSV file exists
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"The file {PATH} does not exist.")

    # Read only the id and abstract columns
    df = pd.read_csv(PATH)[['id', 'abstract']]
    count_titles = 2
    titles_list = []

    # Iterate over each subgraph to generate summaries
    for i, subgraph in enumerate(subgraphs, start=1):
        # Extract the color attribute from the first node
        color = list(subgraph.nodes(data=True))[0][1]['color']
        num_nodes = len(subgraph.nodes())
        # Skip clusters with only one node
        if num_nodes == 1:
            continue

        # Extract abstracts corresponding to the nodes in the subgraph
        node_ids = set(subgraph.nodes())
        abstracts = df[df['id'].isin(node_ids)]['abstract'].dropna().tolist()

        # If no abstracts are found, skip the cluster
        if not abstracts:
            print(f"Cluster {i}: No abstracts found. Skipping.")
            continue

        # Combine all abstracts into a single text block with clear delimiters and instructional prompt
        combined_abstracts = " ".join([f"<New Text:> {abstract}" for j, abstract in enumerate(abstracts)])
        with open('prompt for command-r.txt', 'r') as file:
            instructions_command_r = file.read()
        # Display summarization information
        print(f"Summarizing {len(abstracts)} abstracts.\nCluster color: {color}.\nNumber of vertices: {num_nodes}.")

        # Generate the summary using Cohere's summarize API
        response = co.generate(
                model='command-r-plus-08-2024',
                prompt=instructions_command_r+combined_abstracts,
                max_tokens=300
        )
        summary = response.generations[0].text.strip()

        with open('prompt for llama.txt', 'r') as file:
            instructions_llama = file.read()
        input_params = {
            "prompt": instructions_llama + summary,
            "max_tokens": 300
        }
        output = replicate.run(
            "meta/meta-llama-3.1-405b-instruct",
            input=input_params
        )
        summary = "".join(output)

        with open('prompt for llama 2.txt' , 'r') as file:
            instructions_llama2 = file.read()
        input_params = {
            "prompt": instructions_llama2 + summary,
            "max_tokens": 300
        }
        output = replicate.run(
            "meta/meta-llama-3.1-405b-instruct",
            input=input_params
        )
        title = "".join(output)
        title = title.replace('"', '')
        if title in titles_list:
            title = f"{title} ({count_titles})"
            count_titles += 1
        titles_list.append(title)
        # save the summary.
        if save:

            vers = 'vertices' if num_nodes != 1 else 'vertex'

            file_name = f'{title} ({num_nodes} {vers}).txt'

            try:
                with open(result_file_path + file_name, 'w') as f:
                    f.write(summary)
                    print(f"Summary saved to {result_file_path + file_name}") #
            except FileNotFoundError:  # create the directory if it doesn't exist.
                os.makedirs(result_file_path)
                with open(file_name, 'w') as f:
                    f.write(summary)
        print('\n*3')
    return titles_list #
        