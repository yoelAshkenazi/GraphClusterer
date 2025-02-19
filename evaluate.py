"""
Yoel Ashkenazi
This file is responsible for evaluating the summarization results using Cohere's API.
"""
import pandas as pd
import random
import numpy as np
import networkx as nx
import cohere  # Added Cohere import
import os
from dotenv import load_dotenv
from requests.exceptions import Timeout  # Import Timeout exception
import time  # For optional delays between retries


load_dotenv()
cohere_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client with your API key
# co = cohere.Client(api_key=cohere_key)

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
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant 
information it contains.
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
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points 
of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on 
the Evaluation Criteria.
"""

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not 
supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

FLUENCY_SCORE_CRITERIA = """
Fluency(1-5): the quality of the summary in terms of grammar, spelling, punctuation, word choice, 
and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still 
comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
4: Very Good. The summary is nearly flawless with minor errors that do not impede understanding.
5: Excellent. The summary is completely fluent with perfect grammar, spelling, punctuation, and sentence structure.
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


def generate_with_retry(co_client, model, prompt, max_tokens=100, temperature=0.0, retries=3, delay=2):
    """
    Generate a response from Cohere's API with retry on Timeout.
    
    :param co_client: Cohere client instance.
    :param model: The Cohere model to use.
    :param prompt: The prompt to send to the model.
    :param max_tokens: Maximum tokens in the response.
    :param temperature: Sampling temperature.
    :param retries: Number of retry attempts.
    :param delay: Delay in seconds between retries.
    :return: The response from Cohere or None if all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            response = co_client.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Timeout:
            print(f"Timeout occurred for prompt. Retrying ({attempt}/{retries})...")
            if attempt < retries:
                time.sleep(delay)  # Optional: wait before retrying
            else:
                print("Max retries reached. Moving on.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None


def metrics_evaluations(name: str, vertices: pd.DataFrame, G: nx.Graph = None):
    """
    Load the cluster summary for the given name and calculate average scores for relevancy, coherence, consistency,
    and fluency.
    :param name: the name of the dataset.
    :param vertices: the vertices DataFrame.
    :param G: the graph.
    :return: Averages of the averages for relevancy, coherence, consistency, and fluency.
    """
    summary_path = f"Results/Summaries/{name}/"

    # Check if the summary directory exists and list its contents
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Directory '{summary_path}' does not exist.")

    clusters = os.listdir(summary_path)
    if not clusters:
        raise FileNotFoundError(f"No summary files found in directory {summary_path}.")

    # Prepare to store summaries and subgraphs
    summaries = {}
    titles = [cluster.split('.')[0] for cluster in clusters]  # Get the titles.
    title_to_color = extract_colors(G)
    subgraphs = {}

    # Filter out non-paper nodes
    articles = [node for node in G.nodes if G.nodes[node].get('type', '') == 'paper']
    G = G.subgraph(articles)

    for i, title in enumerate(titles):
        decode_break = False
        summary_file_path = os.path.join(summary_path, clusters[i])
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            try:
                summaries[title] = f.read()
            except UnicodeDecodeError:
                decode_break = True
        if decode_break:
            print(f"Failed to decode summary file: {summary_file_path}")
            continue
        # Get the subgraph.
        color = title_to_color.get(title, 'green')  # Default color if not found
        nodes = [node for node in G.nodes if G.nodes[node]['color'] == color]

        subgraphs[title] = G.subgraph(nodes)

    data = vertices[['id', 'abstract']]  # Read only the id and abstract columns

    # Initialize lists to store the average scores for each cluster
    all_relevancy_scores = {title: 0 for title in titles}
    all_coherence_scores = {title: 0 for title in titles}
    all_consistency_scores = {title: 0 for title in titles}
    all_fluency_scores = {title: 0 for title in titles}

    for title, summary in summaries.items():  # Iterate through each cluster

        # Initialize Cohere client with your API key
        co = cohere.Client(api_key=cohere_key)

        subgraph = subgraphs.get(title, nx.Graph())
        cluster_name = title

        cluster_abstracts = [abstract for id_, abstract in data.values if id_ in subgraph.nodes()]
        # Clean NaNs
        cluster_abstracts = [abstract for abstract in cluster_abstracts if not pd.isna(abstract)]
        cluster_sample_size = max(1, int(0.2 * len(cluster_abstracts)))

        if len(cluster_abstracts) < cluster_sample_size:
            print(f"Warning: Adjusting sample size for cluster '{cluster_name}' due to small population.")
            cluster_sample_size = len(cluster_abstracts)

        if len(cluster_abstracts) == 0:
            print(f"No abstracts found in cluster '{cluster_name}'. Skipping this cluster.")
            continue

        try:
            cluster_abstracts_sampled = random.sample(cluster_abstracts, cluster_sample_size)
        except ValueError as e:
            print(f"Sampling error for cluster '{cluster_name}': {e}")
            cluster_abstracts_sampled = cluster_abstracts

        cluster_relevancy_scores = []
        cluster_coherence_scores = []
        cluster_consistency_scores = []
        cluster_fluency_scores = []

        for abstract in cluster_abstracts_sampled:
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

                # Call Cohere's API with retry
                response = generate_with_retry(
                    co_client=co,
                    model="command-r-plus-08-2024",  # Replace with the desired Cohere model
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.0,
                    retries=3,
                    delay=2
                )

                if response is None:
                    print(f"Failed to get response for metric '{eval_type}' on cluster '{cluster_name}'. "
                          f"Assigning score 0.")
                    score = 0
                else:
                    # Extract the response text
                    score_text = response.generations[0].text.strip()

                    # Process and store the score in the corresponding list
                    if score_text.endswith('.'):
                        score_text = score_text[:-1]
                    try:
                        score = int(score_text[-1])
                        if not 1 <= score <= 5:
                            raise ValueError("Score out of expected range.")
                    except (IndexError, ValueError):
                        print(f"Unexpected score format: '{score_text}'. Defaulting to 0.")
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
        avg_cluster_relevancy = (sum(cluster_relevancy_scores) / len(cluster_relevancy_scores)) \
            if cluster_relevancy_scores else 0
        avg_cluster_coherence = (sum(cluster_coherence_scores) / len(cluster_coherence_scores)) \
            if cluster_coherence_scores else 0
        avg_cluster_consistency = (sum(cluster_consistency_scores) / len(cluster_consistency_scores)) \
            if cluster_consistency_scores else 0
        avg_cluster_fluency = (sum(cluster_fluency_scores) / len(cluster_fluency_scores)) \
            if cluster_fluency_scores else 0

        # Store these averages to later calculate the dataset-level averages.
        all_relevancy_scores[title] = avg_cluster_relevancy / 5
        all_coherence_scores[title] = avg_cluster_coherence / 5
        all_consistency_scores[title] = avg_cluster_consistency / 5
        all_fluency_scores[title] = avg_cluster_fluency / 5

    # Calculate the overall averages across all clusters.
    avg_relevancy = max(0, np.mean(list(all_relevancy_scores.values())))
    avg_coherence = max(0, np.mean(list(all_coherence_scores.values())))
    avg_consistency = max(0, np.mean(list(all_consistency_scores.values())))
    avg_fluency = max(0, np.mean(list(all_fluency_scores.values())))

    # Return an additional dictionary with the cluster-level averages ({title: [scores]})
    titles_to_scores = {
        title: [all_relevancy_scores[title], all_coherence_scores[title], all_consistency_scores[title],
                all_fluency_scores[title]] for title in titles
    }

    return avg_relevancy, avg_coherence, avg_consistency, avg_fluency, titles_to_scores


def extract_colors(graph: nx.Graph) -> dict:
    """
    Extract the colors of the clusters from the graph.
    :param graph: the graph.
    :return:
    """
    title_to_color = {}
    for node in graph.nodes:
        title = graph.nodes[node].get('title', None)
        if title is not None:
            title_to_color[title] = graph.nodes[node].get('color', 'green')  # Default to 'green' if color not set
    return title_to_color


def evaluate(name: str, vertices, G: nx.Graph = None) -> float:
    """
    Load the cluster summary for the given name and evaluate consistency.
    :param name: the name of the dataset.
    :param vertices: the vertices DataFrame.
    :param G: the graph.
    :return: A float representing the evaluation score.
    """
    summary_path = f"Results/Summaries/{name}/"

    # Check if the summary directory exists
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Directory '{summary_path}' does not exist.")

    clusters = os.listdir(summary_path)
    if not clusters:
        raise FileNotFoundError(f"No summary files found in directory {summary_path}.")

    summaries = {}
    titles = [cluster.split('.')[0] for cluster in clusters]  # Get the titles.
    title_to_color = extract_colors(G)
    subgraphs = {}

    for i, title in enumerate(titles):  # Iterate through each cluster
        summary_file_path = os.path.join(summary_path, clusters[i])
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                summaries[title] = f.read()
        except UnicodeDecodeError:
            print(f"Failed to decode summary file: {summary_file_path}")
            continue

        # Get the subgraph.
        color = title_to_color.get(title, 'green')  # Default color if not found
        nodes = [node for node in G.nodes if G.nodes[node].get('color', 'green') == color]
        subgraphs[title] = G.subgraph(nodes)

    # Determine the path to the CSV file based on the source
    data = vertices[['id', 'abstract']]  # Read only the id and abstract columns

    evaluations = {}
    total_in_score = 0  # Total scores for the abstracts sampled inside the clusters.
    total_out_score = 0  # Total scores for the abstracts sampled outside the clusters.

    for title, summary in summaries.items():

        # Initialize Cohere client with your API key
        co = cohere.Client(api_key=cohere_key)

        # Get the subgraph.
        subgraph = subgraphs.get(title, nx.Graph())
        cluster_name = title

        # Get the abstracts from the cluster.
        cluster_abstracts = [abstract for id_, abstract in data.values if id_ in subgraph.nodes()]
        # Clean NaNs.
        cluster_abstracts = [abstract for abstract in cluster_abstracts if not pd.isna(abstract)]
        cluster_sample_size = max(1, int(0.2 * len(cluster_abstracts)))

        if len(cluster_abstracts) < cluster_sample_size:
            print(f"Warning: Adjusting sample size for cluster '{cluster_name}' due to small population.")
            cluster_sample_size = len(cluster_abstracts)

        if len(cluster_abstracts) == 0:
            print(f"No abstracts found in cluster '{cluster_name}'. Skipping this cluster.")
            continue

        try:
            cluster_abstracts_sampled = random.sample(cluster_abstracts, cluster_sample_size)
        except ValueError as e:
            print(f"Sampling error for cluster '{cluster_name}': {e}")
            cluster_abstracts_sampled = cluster_abstracts

        # Get the abstracts from outside the cluster.
        outside_abstracts = [abstract for id_, abstract in data.values if id_ not in subgraph.nodes()]
        # Clean NaNs.
        outside_abstracts = [abstract for abstract in outside_abstracts if not pd.isna(abstract)]

        # Ensure that the sample size does not exceed the number of available abstracts
        if len(outside_abstracts) < cluster_sample_size:
            print(f"Warning: Adjusting sample size for outside cluster '{cluster_name}' due to small population.")
            outside_sample_size = len(outside_abstracts)
        else:
            outside_sample_size = cluster_sample_size

        if len(outside_abstracts) == 0:
            print(f"No outside abstracts found for cluster '{cluster_name}'. Skipping outside evaluations.")
            continue

        try:
            outside_abstracts_sampled = random.sample(outside_abstracts, outside_sample_size)
        except ValueError as e:
            print(f"Sampling error for outside cluster '{cluster_name}': {e}")
            outside_abstracts_sampled = outside_abstracts

        # Initialize current scores for this cluster
        current_score_in = 0
        current_score_out = 0

        # Determine the number of pairs to evaluate
        num_pairs = min(len(cluster_abstracts_sampled), len(outside_abstracts_sampled))

        for i in range(num_pairs):
            # Evaluate consistency for abstracts inside the cluster.
            prompt_in = (
                f"Answer using only a number between 1 to 100: "
                f"How consistent is the following summary with the abstract?\n"
                f"Summary: {summary}\n"
                f"Abstract: {cluster_abstracts_sampled[i]}\n\n\n"
                f"No matter what is the score you choose to give, do not explain why you gave this score."
            )

            response_in = generate_with_retry(
                co_client=co,
                model="command-r-plus-08-2024",
                prompt=prompt_in,
                max_tokens=100,
                temperature=0.0,
                retries=3,
                delay=2
            )

            if response_in is None:
                print(f"Failed to get response for inside abstract in cluster '{cluster_name}'. Assigning score 0.")
                score_in = 0
            else:
                # Extract the response text
                score_in_text = response_in.generations[0].text.strip()
                try:
                    score_in = int(score_in_text.split('\n')[-1].split(':')[-1])
                    if not 1 <= score_in <= 100:
                        raise ValueError("Score out of expected range.")
                except (IndexError, ValueError):
                    print(f"Unexpected score format: '{score_in_text}'. Defaulting to 0.")
                    score_in = 0

            # Evaluate consistency for abstracts outside the cluster.
            prompt_out = (
                f"Answer using only a number between 0 to 100: "
                f"How consistent is the following summary with the abstract?\n"
                f"Summary: {summary}\n"
                f"Abstract: {outside_abstracts_sampled[i]}\n\n\n"
                f"Even if the summary is not consistent with the abstract, please provide a score between "
                f"0 to 100, and only the score."
            )

            response_out = generate_with_retry(
                co_client=co,
                model="command-r-plus-08-2024",  # Replace with the desired Cohere model
                prompt=prompt_out,
                max_tokens=100,
                temperature=0.0,
                retries=3,
                delay=2
            )

            if response_out is None:
                print(f"Failed to get response for outside abstract in cluster '{cluster_name}'. Assigning score 0.")
                score_out = 0
            else:
                # Extract the response text
                score_out_text = response_out.generations[0].text.strip()
                try:
                    score_out = int(score_out_text.split('\n')[-1].split(':')[-1])
                    if not 0 <= score_out <= 100:
                        raise ValueError("Score out of expected range.")
                except (IndexError, ValueError):
                    print(f"Unexpected score format: '{score_out_text}'. Defaulting to 0.")
                    score_out = 0

            # Accumulate the scores for this iteration
            current_score_in += score_in
            current_score_out += score_out

        # After processing all pairs for this cluster, add to total scores
        total_in_score += current_score_in
        total_out_score += current_score_out

        # Determine the decision based on total scores for this cluster
        decision = "consistent" if current_score_in >= current_score_out else "inconsistent"

        evaluations[cluster_name] = (current_score_in, current_score_out, decision)

        # Print the summary for this cluster
        print(f"Cluster summary for cluster '{cluster_name}' is {decision} with the cluster abstracts. "
              f"\nScore in: {current_score_in}\nScore out: {current_score_out}\n{'-' * 50}")

    # Calculate the final evaluation score
    if (total_in_score + total_out_score) != 0:
        final_score = total_in_score / (total_in_score + total_out_score)
    else:
        final_score = 0

    return final_score
