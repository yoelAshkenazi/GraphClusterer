"""
Module: collect_api_responses.py

This script is responsible for evaluating summarization results using Cohere's API.
It collects the raw responses from the API for further analysis.

Usage:
    python collect_api_responses.py
"""

import os
import pandas as pd
import numpy as np
import random
import pickle as pkl
import networkx as nx
from typing import List
import json
import logging
import cohere  # Cohere's API
from dotenv import load_dotenv  # Only if using .env for API keys


# ----------------------------- #
#        Configuration          #
# ----------------------------- #

# Load environment variables from .env file if present
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Cohere client with your API key
COHERE_API_KEY = os.getenv("api_key")
if not COHERE_API_KEY:
    logging.error("Cohere API key not found. Please set the COHERE_API_KEY environment variable.")
    exit(1)

co = cohere.Client(api_key=COHERE_API_KEY)  # Initialize Cohere client

# Define the Evaluation Prompt Template
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.
Your answer will only be an integer in the range 1-5.
Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Define the Criteria and Steps for each Metric
RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

FLUENCY_SCORE_CRITERIA = """
Fluency(1-5): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 5.
"""

# Define the evaluation metrics dictionary
evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

# ----------------------------- #
#          Functions             #
# ----------------------------- #

def load_graph(name: str, version: str, proportion: float, k: int = 5, weight: float = 1, optimized: bool = False) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: Name of the graph/dataset.
    :param version: Version of the graph ('distances', 'original', 'proportion').
    :param proportion: Proportion parameter for the graph.
    :param k: KNN parameter.
    :param weight: Weight of the edges.
    :param optimized: Flag to use optimized graph version.
    :return: Loaded NetworkX graph.
    """
    if optimized:
        graph_path = None
        optimized_graphs_dir = 'data/optimized_graphs'
        for file in os.listdir(optimized_graphs_dir):
            if file.startswith(name):
                graph_path = os.path.join(optimized_graphs_dir, file)
                break
        if not graph_path:
            logging.error(f"No optimized graph found for dataset '{name}' in '{optimized_graphs_dir}'.")
            exit(1)
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', or 'proportion'."
        graph_path = f"data/processed_graphs/k_{k}/"
        if weight != 1:
            graph_path += f"weight_{weight}/"
        if version == 'distances':
            graph_path += f"only_distances/{name}.gpickle"
        elif version == 'original':
            graph_path += f"only_original/{name}.gpickle"
        else:  # 'proportion'
            if proportion != 0.5:
                graph_path += f"{name}_proportion_{proportion}.gpickle"
            else:
                graph_path += f"{name}.gpickle"

    # Load the graph
    if not os.path.exists(graph_path):
        logging.error(f"Graph file not found at path: {graph_path}")
        exit(1)

    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # Filter the graph to remove NaN nodes
    nodes = list(graph.nodes(data=True))
    initial_node_count = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    removed_nodes = initial_node_count - len(nodes)
    logging.info(f"Successfully removed {removed_nodes} {'node' if removed_nodes == 1 else 'nodes'} from the graph.")
    graph = graph.subgraph(nodes).copy()

    return graph


def filter_by_colors(graph: nx.Graph) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to community colors.
    :param graph: The NetworkX graph.
    :return: List of subgraphs partitioned by color.
    """
    # Filter articles by vertex type
    articles = [node for node in graph.nodes() if graph.nodes.data()[node].get('type') == 'paper']
    articles_graph = graph.subgraph(articles).copy()
    graph = articles_graph

    # Partition the graph by colors
    nodes = list(graph.nodes(data=True))
    colors = set([data.get('color') for node, data in nodes])
    subgraphs = []
    for color in colors:
        color_nodes = [node for node, data in nodes if data.get('color') == color]
        subgraph = graph.subgraph(color_nodes).copy()
        subgraphs.append(subgraph)
        logging.info(f"Created subgraph for color '{color}' with {len(subgraph.nodes())} nodes.")

    return subgraphs


def collect_api_responses(
    name: str,
    version: str,
    proportion: float = 0.5,
    k: int = 5,
    weight: float = 1,
    optimized: bool = False,
    output_file: str = "api_responses.json"
):
    """
    Collect raw API responses from Cohere's API for evaluation prompts.
    :param name: Name of the dataset.
    :param version: Version of the graph ('distances', 'original', 'proportion').
    :param proportion: Proportion parameter for the graph.
    :param k: KNN parameter.
    :param weight: Weight of the edges.
    :param optimized: Flag to use optimized graph version.
    :param output_file: Path to the output JSON file to save responses.
    :return: None
    """
    # Determine the summary path
    if optimized:
        summary_dir = "Summaries/optimized/"
        summary_files = [file for file in os.listdir(summary_dir) if file.startswith(name)]
        if not summary_files:
            logging.error(f"No optimized summaries found for dataset '{name}' in '{summary_dir}'.")
            exit(1)
        summary_path = os.path.join(summary_dir, summary_files[0])
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', or 'proportion'."
        summary_dir = f"Summaries/k_{k}/"
        if weight != 1:
            summary_dir += f"weight_{weight}/"
        if version == 'distances':
            summary_dir += f"{name}_only_distances/"
        elif version == 'original':
            summary_dir += f"{name}_only_original/"
        else:  # 'proportion'
            if proportion != 0.5:
                summary_dir += f"{name}_proportion_{proportion}/"
            else:
                summary_dir += f"{name}/"

        if not os.path.exists(summary_dir):
            logging.error(f"Summary directory not found at path: {summary_dir}")
            exit(1)

    # Load the graph
    graph = load_graph(name, version, proportion, k, weight, optimized)
    logging.info(f"Loaded graph: {graph}")

    # Prepare to store summaries and subgraphs
    summaries = {}
    clusters = os.listdir(summary_dir)
    if not clusters:
        logging.error(f"No summary files found in directory: {summary_dir}")
        exit(1)
    colors = [cluster.split('_')[2] for cluster in clusters]  # Assumes filenames are formatted accordingly
    subgraphs = {}

    for i, cluster in enumerate(clusters):
        summary_file_path = os.path.join(summary_dir, cluster)
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                summaries[cluster] = f.read()
        except UnicodeDecodeError:
            logging.warning(f"UnicodeDecodeError while reading file: {summary_file_path}. Skipping.")
            continue

        # Get the subgraph based on color
        color = colors[i]
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node].get('color') == color]
        subgraph = graph.subgraph(nodes).copy()
        subgraphs[cluster] = subgraph
        logging.info(f"Processed subgraph for cluster '{cluster}' with color '{color}' containing {len(subgraph.nodes())} nodes.")

    # Load the data
    data_csv_path = f"data/graphs/{name}_papers.csv"
    if not os.path.exists(data_csv_path):
        logging.error(f"Data CSV file not found at path: {data_csv_path}")
        exit(1)
    data = pd.read_csv(data_csv_path)[['id', 'abstract']]

    # Prepare to store API responses
    responses_data = []

    # Iterate through each cluster and its summary
    for cluster, summary in summaries.items():
        subgraph = subgraphs[cluster]
        cluster_name = cluster.split('_')[2]  # Assumes filenames are formatted accordingly

        # Get abstracts from the cluster
        cluster_ids = list(subgraph.nodes())
        cluster_abstracts = data[data['id'].isin(cluster_ids)]['abstract'].dropna().tolist()

        # Get abstracts outside the cluster
        outside_abstracts = data[~data['id'].isin(cluster_ids)]['abstract'].dropna().tolist()

        # Sample 20% of the cluster size for evaluation
        cluster_sample_size = max(1, int(0.2 * len(cluster_abstracts)))
        sampled_cluster_abstracts = random.sample(cluster_abstracts, cluster_sample_size)
        try:
            sampled_outside_abstracts = random.sample(outside_abstracts, cluster_sample_size)
        except ValueError:
            sampled_outside_abstracts = random.sample(outside_abstracts, len(outside_abstracts))  # Sample all if insufficient

        # Iterate through sampled abstracts
        for i in range(min(len(sampled_cluster_abstracts), len(sampled_outside_abstracts))):
            abstract_in = sampled_cluster_abstracts[i]
            abstract_out = sampled_outside_abstracts[i]

            # Define pairs for evaluation
            abstract_pairs = [
                ("in_cluster", abstract_in),
                ("out_cluster", abstract_out)
            ]

            for abstract_type, abstract in abstract_pairs:
                for eval_type, (criteria, steps) in evaluation_metrics.items():
                    # Generate the evaluation prompt
                    prompt = EVALUATION_PROMPT_TEMPLATE.format(
                        criteria=criteria,
                        steps=steps,
                        metric_name=eval_type,
                        document=abstract,
                        summary=summary,
                    )

                    # Send prompt to Cohere's API
                    try:
                        response = co.chat(
                            model="command-r-plus-08-2024",  # Replace with the desired Cohere model
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=100,
                            temperature=0.0  # Set temperature to 0 for deterministic output
                        )
                        response_text = response.generations[0].text.strip()
                    except Exception as e:
                        logging.error(f"Error during API call: {e}")
                        response_text = ""

                    # Store the response
                    responses_data.append({
                        "cluster": cluster_name,
                        "metric": eval_type,
                        "abstract_type": abstract_type,
                        "abstract": abstract,
                        "summary": summary,
                        "prompt": prompt,
                        "response": response_text
                    })

                    logging.info(f"Collected response for cluster '{cluster_name}', metric '{eval_type}', abstract type '{abstract_type}'.")

    # Save the collected responses to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, ensure_ascii=False, indent=4)

    logging.info(f"All API responses have been saved to '{output_file}'.")


# ----------------------------- #
#          Main Block            #
# ----------------------------- #

if __name__ == "__main__":
    # Example parameters (replace with actual values as needed)
    collect_api_responses(
        name="3D printing",          # Replace with your dataset name
        version="original",                # 'distances', 'original', or 'proportion'
        proportion=0.5,                    # Relevant if version='proportion'
        k=5,                               # Relevant if version involves KNN
        weight=1,                          # Edge weight parameter
        optimized=False,                   # Set to True if using optimized graphs/summaries
        output_file="responses.json"   # Output file to save responses
    )
