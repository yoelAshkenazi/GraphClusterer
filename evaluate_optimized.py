"""
Yoel Ashkenazi
This file is responsible for evaluating the summarization results using Cohere's API.
"""
import pandas as pd
import random
import pickle as pkl
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
import cohere  # Added Cohere import
import os


# Initialize Cohere client with your API key
co = cohere.Client(api_key="nECzpidx1wy8QavqvK3CUxOQWNjMJVnoKNI26wez")

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


def myEval(name: str, version: str, proportion: float = 0.5, k: int = 5, weight: float = 1, optimized: bool = False):
    """
    Load the cluster summary for the given name and calculate average scores for relevancy, coherence, consistency,
    and fluency.
    :param name: the name of the dataset.
    :param version: the version of the graph.
    :param proportion: the proportion of the graph to use.
    :param k: The KNN parameter.
    :param weight: the weight of the edges.
    :param optimized: whether to use the optimized version of the graph.
    :return: Averages of the averages for relevancy, coherence, consistency, and fluency.
    """
    if optimized:
        summary_path = f"Summaries/optimized/"
        for file in os.listdir('Summaries/optimized'):
            if file.startswith(name):
                summary_path += file
                break
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', " \
                                                                   "or 'proportion'."
        summary_path = f"Summaries/k_{k}/"
        if weight != 1:
            summary_path += f"weight_{weight}/"
        if version == 'distances':
            summary_path += f"{name}_only_distances/"

        elif version == 'original':
            summary_path += f"{name}_only_original/"

        else:
            if proportion != 0.5:
                summary_path += f"{name}_proportion_{proportion}/"
            else:
                summary_path += f"{name}/"

    # Load the graph
    graph = load_graph(name, version, proportion, k, weight, optimized)
    G = graph
    print(G)  # Print the graph.

    # Prepare to store summaries and subgraphs
    summaries = {}
    clusters = os.listdir(summary_path)
    colors = [cluster.split('_')[2] for cluster in clusters]  # Get the colors.
    subgraphs = {}

    for i, cluster in enumerate(clusters):
        decode_break = False
        with open(os.path.join(summary_path, cluster), 'r') as f:
            try:
                summaries[cluster] = f.read()
            except UnicodeDecodeError:
                decode_break = True
        if decode_break:
            continue

        # Get the subgraph.
        color = colors[i]
        nodes = [node for node in G.nodes() if G.nodes.data()[node]['color'] == color]
        subgraphs[cluster] = G.subgraph(nodes)
        print(f"{subgraphs[cluster]} (color {color})")  # Print the subgraph.

    # Load the data
    data = pd.read_csv(f"data/graphs/{name}_papers.csv")[['id', 'abstract']]

    all_relevancy_scores = []
    all_coherence_scores = []
    all_consistency_scores = []
    all_fluency_scores = []

    for cluster, summary in summaries.items():
        subgraph = subgraphs[cluster]
        cluster_name = cluster.split('_')[2]  # Get the color.

        # Get the abstracts from the cluster.
        cluster_abstracts = [abstract for id_, abstract in data.values if id_ in subgraph.nodes()]
        

        # Clean NaNs and sample 20% of the cluster size.
        cluster_abstracts = [abstract for abstract in cluster_abstracts if not pd.isna(abstract)]
        sampled_abstracts = random.sample(cluster_abstracts, max(1, int(0.2 * len(cluster_abstracts))))

        cluster_relevancy_scores = []
        cluster_coherence_scores = []
        cluster_consistency_scores = []
        cluster_fluency_scores = []

        for abstract in sampled_abstracts:
            # Iterate through each metric for evaluation
            for eval_type, (criteria, steps) in evaluation_metrics.items():
                # Generate the evaluation prompt
                prompt = EVALUATION_PROMPT_TEMPLATE.format(
                    criteria=criteria,
                    steps=steps,
                    metric_name=eval_type,
                    document=abstract,
                    summary=summary,
                )

                # Call Cohere's API to evaluate the metric
                response = co.generate(
                    model="command-r-plus-08-2024",  # Replace with the desired Cohere model
                    prompt=prompt,  # Use the prompt directly instead of messages
                    max_tokens=100,
                    temperature=0.0  # Set temperature to 0 for deterministic output
                )

                # Extract the response text
                score = response.generations[0].text.strip()

                # Process and store the score in the corresponding list
                if score.endswith('.'):
                    score = score[:-1]
                try:
                    score = int(score[-1])
                except (IndexError, ValueError):
                    print(f"Unexpected score format: '{score}'. Defaulting to 0.")
                    score = 0

                if eval_type == "Relevance":
                    cluster_relevancy_scores.append(score)
                elif eval_type == "Coherence":
                    cluster_coherence_scores.append(score)
                elif eval_type == "Consistency":
                    cluster_consistency_scores.append(score)
                elif eval_type == "Fluency":
                    cluster_fluency_scores.append(score)

        # Calculate average for each cluster.
        avg_cluster_relevancy = sum(cluster_relevancy_scores) / len(cluster_relevancy_scores) if cluster_relevancy_scores else 0
        avg_cluster_coherence = sum(cluster_coherence_scores) / len(cluster_coherence_scores) if cluster_coherence_scores else 0
        avg_cluster_consistency = sum(cluster_consistency_scores) / len(cluster_consistency_scores) if cluster_consistency_scores else 0
        avg_cluster_fluency = sum(cluster_fluency_scores) / len(cluster_fluency_scores) if cluster_fluency_scores else 0

        # Store these averages to later calculate the dataset-level averages.
        all_relevancy_scores.append(avg_cluster_relevancy)
        all_coherence_scores.append(avg_cluster_coherence)
        all_consistency_scores.append(avg_cluster_consistency)
        all_fluency_scores.append(avg_cluster_fluency)

    # Calculate the overall averages across all clusters.
    avg_relevancy = sum(all_relevancy_scores) / len(all_relevancy_scores) if all_relevancy_scores else 0
    avg_coherence = sum(all_coherence_scores) / len(all_coherence_scores) if all_coherence_scores else 0
    avg_consistency = sum(all_consistency_scores) / len(all_consistency_scores) if all_consistency_scores else 0
    avg_fluency = sum(all_fluency_scores) / len(all_fluency_scores) if all_fluency_scores else 0

    # Print the scores
    print("Relevance: ", avg_relevancy)
    print("Coherence: ", avg_coherence)
    print("Consistency: ", avg_consistency)
    print("Fluency: ", avg_fluency)
    print('-' * 50)

    return avg_relevancy, avg_coherence, avg_consistency, avg_fluency


def evaluate(name: str, version: str, proportion: float = 0.5, k: int = 5, weight: float = 1, optimized: bool = False):
    """
    Load the cluster summary for the given name.
    :param name: the name of the dataset.
    :param version: the version of the graph.
    :param proportion: the proportion of the graph to use.
    :param k: The KNN parameter.
    :param weight: the weight of the edges.
    :param optimized: whether to use the optimized version of the graph.
    :return:
    """
    if optimized:
        summary_path = f"Summaries/optimized/"
        print(f"Looking for summaries in optimized directory: {summary_path}")  # Debug statement
        for file in os.listdir('Summaries/optimized'):
            print(f"Found file in optimized directory: {file}")  # Debug statement
            if file.startswith(name):
                summary_path += file
                print(f"Using summary path: {summary_path}")  # Debug statement
                break
    else:
        assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', 'proportion'."
        summary_path = f"Summaries/k_{k}/"
        if weight != 1:
            summary_path += f"weight_{weight}/"
        if version == 'distances':
            summary_path += f"{name}_only_distances/"
        elif version == 'original':
            summary_path += f"{name}_only_original/"
        else:
            if proportion != 0.5:
                summary_path += f"{name}_proportion_{proportion}/"
            else:
                summary_path += f"{name}/"
        print(f"Looking for summaries in directory: {summary_path}")  # Debug statement

    # Check if the summary directory exists and list its contents
    if not os.path.exists(summary_path):
        print(f"Error: Directory {summary_path} does not exist!")
        return None
    
    print(f"Listing files in {summary_path}:")
    print(os.listdir(summary_path))  # Debug statement to list all files in the directory

    # Load the graph.
    graph = load_graph(name, version, proportion, k, weight, optimized)
    G = graph
    print(G)  # Print the graph.

    clusters = os.listdir(summary_path)
    print(f"Clusters found: {clusters}")  # Debug statement to list all clusters

    summaries = {}
    colors = [cluster.split('_')[2] for cluster in clusters]  # Get the colors.
    subgraphs = {}
    for i, cluster in enumerate(clusters):
        decode_break = False
        summary_file_path = os.path.join(summary_path, cluster)
        print(f"Opening summary file: {summary_file_path}")  # Debug statement
        with open(summary_file_path, 'r') as f:
            try:
                summaries[cluster] = f.read()
                print(f"Loaded summary for cluster {cluster}")  # Debug statement
                print(f"Summary: {summaries[cluster]}")  # Debug statement
            except UnicodeDecodeError:
                decode_break = True
        if decode_break:
            print(f"Failed to decode summary file: {summary_file_path}")  # Debug statement
            continue

        # Get the subgraph.
        color = colors[i]
        nodes = [node for node in G.nodes() if G.nodes.data()[node]['color'] == color]
        subgraphs[cluster] = G.subgraph(nodes)
        print(f"{subgraphs[cluster]} (color {color})")  # Print the subgraph.

    # For each summary and cluster pairs, sample abstracts from the cluster and outside the cluster.
    # Then ask Cohere's API which abstracts are more similar to the summary.

    data = pd.read_csv(f"data/graphs/{name}_papers.csv")[['id', 'abstract']]  # Load the data.
    evaluations = {}
    total_in_score = 0  # Total scores for the abstracts sampled inside the clusters.
    total_out_score = 0  # Total scores for the abstracts sampled outside the clusters.
    for cluster, summary in summaries.items():
        # Get the subgraph.
        subgraph = subgraphs[cluster]
        cluster_name = cluster.split('_')[2]  # Get the color.

        # Get the abstracts from the cluster.
        cluster_abstracts = [abstract for id_, abstract in data.values if id_ in subgraph.nodes()]

        # Clean NaNs.
        cluster_abstracts = [abstract for abstract in cluster_abstracts if not pd.isna(abstract)]

        # Determine the sample size, ensuring it's not larger than the available abstracts
        cluster_sample_size = max(1, int(0.2 * len(cluster_abstracts)))
        if len(cluster_abstracts) < cluster_sample_size:
            print(f"Warning: Adjusting sample size for cluster '{cluster_name}' due to small population.")
            cluster_sample_size = len(cluster_abstracts)

        if len(cluster_abstracts) == 0:
            print(f"No abstracts found in cluster '{cluster_name}'. Skipping this cluster.")
            continue

        cluster_abstracts = random.sample(cluster_abstracts, cluster_sample_size)

        # Get the abstracts from outside the cluster.
        outside_abstracts = [abstract for id_, abstract in data.values if id_ not in subgraph.nodes()]

        # Clean NaNs.
        outside_abstracts = [abstract for abstract in outside_abstracts if not pd.isna(abstract)]

        # Ensure that the sample size does not exceed the number of available abstracts
        if len(outside_abstracts) < cluster_sample_size:
            print(f"Warning: Adjusting sample size for outside cluster '{cluster_name}' due to small population.")
            cluster_sample_size = len(outside_abstracts)

        outside_abstracts = random.sample(outside_abstracts, cluster_sample_size)

        # Ask Cohere's API which abstracts are more similar to the summary.
        for i in range(min(len(cluster_abstracts), len(outside_abstracts))):
            # Evaluate consistency for abstracts inside the cluster.
            prompt_in = f"Answer using only a number between 1 to 100: " \
                        f"How consistent is the following summary with the abstract?\n" \
                        f"Summary: {summary}\n" \
                        f"Abstract: {cluster_abstracts[i]}\n"

            response_in = co.generate (
                model="command-r-plus-08-2024",
                prompt=prompt_in,
                max_tokens=100,
                temperature=0.0
            )

            score_in_text = response_in.generations[0].text.strip()
            try:
                score_in = int(score_in_text.split('\n')[-1].split(':')[-1])
            except (IndexError, ValueError):
                print(f"Unexpected score format: '{score_in_text}'. Defaulting to 0.")
                score_in = 0
            total_in_score += score_in

            # Evaluate consistency for abstracts outside the cluster.
            prompt_out = f"Answer using only a number between 1 to 100: " \
                         f"How consistent is the following summary with the abstract?\n" \
                         f"Summary: {summary}\n" \
                         f"Abstract: {outside_abstracts[i]}\n"

            response_out = co.generate(
                model="command-r-plus-08-2024",  # Replace with the desired Cohere model
                prompt=prompt_out,
                max_tokens=100,
                temperature=0.0  # Set temperature to 0 for deterministic output
            )

            score_out_text = response_out.generations[0].text.strip()
            try:
                score_out = int(score_out_text.split('\n')[-1].split(':')[-1])
            except (IndexError, ValueError):
                print(f"Unexpected score format: '{score_out_text}'. Defaulting to 0.")
                score_out = 0
            total_out_score += score_out

        decision = "consistent" if total_in_score >= total_out_score else "inconsistent"

        evaluations[cluster_name] = (total_in_score, total_out_score, decision)

        print(f"Cluster summary for cluster '{cluster_name}' is {decision} with the cluster abstracts. "
              f"\nScore in: {total_in_score}\nScore out: {total_out_score}\n{'-' * 50}")

    return total_in_score / (total_in_score + total_out_score) if (total_in_score + total_out_score) != 0 else 0



def filter_by_colors(graph: nx.Graph) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to the community colors.
    :param graph: the graph.
    :return:
    """
    # First, filter articles by vertex type.
    articles = [node for node in graph.nodes() if graph.nodes.data()[node]['type'] == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # Second filter divides the graph by colors.
    nodes = graph.nodes(data=True)
    colors = set([node[1]['color'] for node in nodes])
    subgraphs = []
    for i, color in enumerate(colors):  # Filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]

        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)

    return subgraphs


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
                graph_path += f"{name}_proportion_{proportion}.gpickle"
            else:
                graph_path += f"{name}.gpickle"

    # Load the graph.
    print(graph_path)
    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # Filter the graph to remove NaN nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    return graph


def plot_bar(name: str, version: str, metrics_dict: list):
    """
    Create and save a bar plot for the given metrics of a specific name and version.

    :param name: the name of the dataset.
    :param version: the version of the graph.
    :param metrics_dict: Dictionary containing metrics for each (name, version) combination.
    :return:

    """
    # Retrieve metrics for the specific name and version
    values = [
        metrics_dict['avg_index'][name][version],  # Average Index
        metrics_dict['largest_cluster_percentage'][name][version],  # Largest Cluster Percentage
        metrics_dict['avg_relevancy'][name][version],  # Average Relevancy
        metrics_dict['avg_coherence'][name][version],  # Average Coherence
        metrics_dict['avg_consistency'][name][version],  # Average Consistency
        metrics_dict['avg_fluency'][name][version],  # Average Fluency
        metrics_dict['success_rates'][name][version]  # Success Rate
    ]

    # Define the labels for the x-axis with line breaks after each word
    x_labels = [
        "Average\nIndex",
        "Largest\nCluster\nPercentage",
        "Average\nRelevancy",
        "Average\nCoherence",
        "Average\nConsistency",
        "Average\nFluency",
        "Success\nRate"
    ]

    # Create the bar plot
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    bars = plt.bar(x_labels, values, color='skyblue', edgecolor='black')

    # Set y-axis limits and labels
    plt.ylim(0, 5.5)  # Adjust according to the expected range of your data
    plt.xlabel("Evaluation Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=14)

    # Set the title of the plot
    plt.title(f"Results for '{name}' with {version} graph", fontsize=16, fontweight='bold')

    # Add gridlines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the value just above the top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,  # Adjust this value to position the text above the bar top
            f"{yval:.2f}",
            ha='center',
            va='bottom',  # Align text to the bottom of the text (above the bar height)
            fontsize=12,
            fontweight='bold'
        )

    # Adjust layout to prevent clipping of labels

    plt.tight_layout()
    plt.show()

    # Save the plot to a file, need to add a filter if it's an optimized graph.
    """
    directory = f"Results/plots/k_{k}/weight_{weight}/"
    plot_file_path = f"{name}"
    if version == 'distances':
        plot_file_path += '_only_distances'
    elif version == 'original':
        plot_file_path += '_only_original'
    else:
        if proportion != 0.5:
            plot_file_path += f'_proportion_{proportion}'
    try:
        plt.savefig(directory + plot_file_path)
    except OSError:
        os.makedirs(directory)
        plt.savefig(directory + plot_file_path)
    """
