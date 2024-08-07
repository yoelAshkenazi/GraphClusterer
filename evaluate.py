"""
Yoel Ashkenazi
This file is responsible for evaluating the summarization results using GPT-4 api.
"""
import os
import pandas as pd
from openai import OpenAI
import random
import pickle as pkl
import networkx as nx
from typing import List

api_key = ('sk-proj-QmQAKfTAciapEYB9EOFC7GzMlxm1Y8OgbpLGaojODhKupIXUVU4A3xO4hMiIpSjwGlm7L27rVmT3BlbkFJvbdo'
           '-Iny4yCtHBN6CZyBBT7IVOY7_x6xPoMKUGtcssmWcezeazQRULrEsG9RpqgegyDdloiQoA')
organization = 'org-FKQBIvqIr7JF5Jhysdnrxx5z'

client = OpenAI(api_key=api_key, organization=organization)


def load_graph(name: str, version: str, proportion, k: int = 5, weight: float = 1) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :param version: the version of the graph.
    :param k: the KNN parameter.
    :param proportion: the proportion of the graph.
    :param weight: the weight of the edges.
    :return:
    """

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

    return subgraphs


def evaluate(name: str, version: str, proportion: float = 0.5, k: int = 5, weight: float = 1):
    """
    Load the cluster summary for the given name.
    :param name: the name of the dataset.
    :param version: the version of the graph.
    :param proportion: the proportion of the graph to use.
    :param k: The KNN parameter.
    :param weight: the weight of the edges.
    :return:
    """
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

    # load the graph.
    graph = load_graph(name, version, proportion, k, weight)
    G = graph
    print(G)  # print the graph.

    clusters = os.listdir(summary_path)

    summaries = {}
    colors = [cluster.split('_')[2] for cluster in clusters]  # get the colors.
    subgraphs = {}
    for i, cluster in enumerate(clusters):
        decode_break = False
        with open(f'{summary_path}{cluster}', 'r') as f:
            try:
                summaries[cluster] = f.read()
            except UnicodeDecodeError:
                decode_break = True
        if decode_break:
            continue

        # get the subgraph.
        color = colors[i]
        nodes = [node for node in G.nodes() if G.nodes.data()[node]['color'] == color]
        subgraphs[cluster] = G.subgraph(nodes)
        print(f"{subgraphs[cluster]} (color {color})")  # print the subgraph.

    # for each summary and cluster pairs, sample abstracts from the cluster and outside the cluster.
    # then ask GPT which abstracts are more similar to the summary.

    data = pd.read_csv(f"data/graphs/{name}_papers.csv")[['id', 'abstract']]  # load the data.
    evaluations = {}
    total_in_score = 0  # total scores for the abstracts sampled inside the clusters.
    total_out_score = 0  # total scores for the abstracts sampled outside the clusters.
    for cluster, summary in summaries.items():
        # get the subgraph.
        subgraph = subgraphs[cluster]
        cluster_name = cluster.split('_')[2]  # get the color.

        # get the abstracts from the cluster.
        cluster_abstracts = [abstract for id_, abstract in data.values if id_ in subgraph.nodes()]

        # get the abstracts from outside the cluster.
        outside_abstracts = [abstract for id_, abstract in data.values if id_ not in subgraph.nodes()]

        # clean nans.
        cluster_abstracts = [abstract for abstract in cluster_abstracts if not pd.isna(abstract)]
        outside_abstracts = [abstract for abstract in outside_abstracts if not pd.isna(abstract)]

        # sample 20% of the cluster size (m) and m abstracts from outside the cluster.
        cluster_abstracts = random.sample(cluster_abstracts, int(0.2 * len(cluster_abstracts)))
        try:
            outside_abstracts = random.sample(outside_abstracts, len(cluster_abstracts))
        except ValueError:
            pass  # if there are not enough abstracts, sample all of them.

        # ask GPT which abstracts are more similar to the summary.
        for i in range(min(len(cluster_abstracts), len(outside_abstracts))):
            # ask GPT which abstract is more similar to the summary.
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Answer using only a number between 1 to 100: "
                                   f"How consistent is the following summary with the abstract?\n"
                                   f"Summary: {summary}\n"
                                   f"Abstract: {cluster_abstracts[i]}\n"
                    }
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5
            )

            score_in = response.choices[0].message.content
            # get the score.
            score_in = int(score_in.split('\n')[-1].split(':')[-1])
            total_in_score += score_in

            # ask GPT which abstract is more similar to the summary.
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Answer using only a number between 1 to 100:"
                                   f"How consistent is the following summary with the abstract?\n"
                                   f"Summary: {summary}\n"
                                   f"Abstract: {outside_abstracts[i]}\n"
                    }
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5
            )

            score_out = response.choices[0].message.content
            # get the score.
            score_out = int(score_out.split('\n')[-1].split(':')[-1])

            total_out_score += score_out

        decision = "consistent" if total_in_score >= total_out_score else "inconsistent"

        evaluations[cluster_name] = (total_in_score, total_out_score, decision)

        print(f"Cluster summary for cluster '{cluster_name}' is {decision} with the cluster abstracts. "
              f"\nScore in: {total_in_score}\nScore out: {total_out_score}\n{'-' * 50}")

    return total_in_score, total_out_score, evaluations
