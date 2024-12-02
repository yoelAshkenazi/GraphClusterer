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

wikipedia = False


def update_wikipedia():
    global wikipedia
    wikipedia = True


def upload_graph(graph: nx.Graph, name: str) -> None:
    """
    Upload the graph to the given path.
    :param graph: the graph to upload.
    :param name: the name of the graph.
    :return: None
    """
    graph_path = f'data/clustered_graphs/{name}.gpickle'

    # save the graph.
    with open(graph_path, 'wb') as f:
        pkl.dump(graph, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"{'-'*50}\nGraph saved to '{graph_path}'.")


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return: the graph.
    :return:
    """
    graph_path = f'data/clustered_graphs/{name}.gpickle'  # the path to the graph.

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
    Partition the graph into subgraphs according to the vertex colors.
    :param graph: the graph.
    :return:
    """
    # Filter non-paper nodes.
    articles = [node for node in graph.nodes if graph.nodes()[node].get('type', '') == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    colors = set([graph.nodes()[n]['color'] for n in graph.nodes])
    subgraphs = []
    for i, color in enumerate(colors):  # filter by colors.
        nodes = [node for node in graph.nodes if graph.nodes()[node]['color'] == color]
        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)
        print(f"color {color} has {len(subgraph.nodes)} vertices.")

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph], name: str, vertices: pd.DataFrame) -> List[str]:
    """
    Summarizes each subgraph's abstract texts using Cohere's API, prints the results, and optionally saves them to
    text files.

    :param subgraphs: List of subgraphs.
    :param name: The name of the dataset.
    :param vertices: The vertices DataFrame.
    :return: None
    """

    result_file_path = f"Results/Summaries/{name}/"  # the path to save the results.

    # Ensure the directory exists
    os.makedirs(result_file_path, exist_ok=True)

    # Clean previous summaries by removing existing files in the directory
    if os.path.exists(result_file_path):
        for file in os.listdir(result_file_path):  # remove all the files in the directory.
            os.remove(os.path.join(result_file_path, file))

    # Integrate dotenv to load environment variables
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)

    # api_token = os.getenv("LLAMA_API_KEY")

    # Load the abstracts from the CSV file
    df = vertices[['id', 'abstract']]
    count_titles = 2
    titles_list = []
    vertex_to_title_map = {}  # map from vertex to title.
    # Iterate over each subgraph to generate summaries
    for i, subgraph in enumerate(subgraphs, start=1):
        # Extract the color attribute from the first node
        color = subgraph.nodes()[list(subgraph.nodes)[0]]['color']

        # Extract abstracts corresponding to the nodes in the subgraph
        node_ids = set(subgraph.nodes())
        abstracts = df[df['id'].isin(node_ids)]['abstract'].dropna().tolist()

        # If only one abstract is present, skip summarization.
        if len(abstracts) <= 1:
            print(f"Cluster {i}: Insufficient abstracts, Skipping.")
            continue

        # Combine all abstracts into a single text block with clear delimiters and instructional prompt
        combined_abstracts = " ".join([f"<New Text:> {abstract}" for j, abstract in enumerate(abstracts)])
        with open('prompt for command-r.txt', 'r') as file:
            instructions_command_r = file.read()
        # Display summarization information
        print(f"{'-'*50}\nSummarizing {len(abstracts)} abstracts.\nCluster color: {color}."
              f"\nNumber of vertices: {len(subgraph)}.")

        # Generate the summary using Cohere's summarize API
        response = co.generate(
                model='command-r-plus-08-2024',
                prompt=instructions_command_r+combined_abstracts,
                max_tokens=300
        )
        summary = response.generations[0].text.strip()

        # Refine the summary by generating a title using the LLAMA API
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
        # title = summary.replace('"', '')

        # Generate a unique title if the title already exists
        with open('prompt for llama 2.txt', 'r') as file:
            instructions_llama2 = file.read()
        prompt = instructions_llama2 + summary
        if titles_list:
            prompt += f"\n\nTry ao avoid giving one of those titles: {titles_list}"
        input_params = {
            "prompt": prompt,
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
        num_nodes = len(subgraph)
        vers = 'vertices' if num_nodes != 1 else 'vertex'

        file_name = f'{title} ({num_nodes} {vers}).txt'

        try:
            with open(result_file_path + file_name, 'w') as f:
                f.write(summary)
                print(f"Summary saved to {result_file_path + file_name}")
        except FileNotFoundError:  # create the directory if it doesn't exist.
            os.makedirs(result_file_path)
            with open(file_name, 'w') as f:
                f.write(summary)
        except UnicodeEncodeError:
            # If the summary contains characters that cannot be encoded, print the summary to the console
            print(f"Summary for {title} ({num_nodes} {vers}) contains characters that cannot be encoded.")
            print(summary)
            exit()  # Exit the program to avoid further errors
        
        for v in subgraph.nodes():
            vertex_to_title_map[v] = f'{title} ({num_nodes} {vers})'

    # Add the titles to the graph
    G = load_graph(name)
    nx.set_node_attributes(G, values=vertex_to_title_map, name='title')
    upload_graph(G, name)
    return titles_list
        