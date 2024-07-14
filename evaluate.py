"""
Yoel Ashkenazi
This file is responsible for evaluating the summarization results using GPT-4 api.
"""
import os
import pandas as pd
import openai
import random
import pickle as pkl


def evaluate(name: str):
    """
    Load the cluster summary for the given name.
    :param name: the name of the dataset.
    :return:
    """

    # load the graph.
    file_path = f"data/processed_graphs/{name}.gpickle"
    with open(file_path, 'rb') as f:
        graph = pkl.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    G = graph
    file_path = f"Summaries/{name}/"

    clusters = os.listdir(file_path)

    summaries = {}
    colors = [cluster.split('_')[2] for cluster in clusters]  # get the colors.
    subgraphs = {}
    for i, cluster in enumerate(clusters):
        with open(f'{file_path}{cluster}', 'r') as f:
            summaries[cluster] = f.read()

        # get the subgraph.
        color = colors[i]
        subgraphs[cluster] = G.subgraph([node for node, data in G.nodes(data=True) if data['color'] == color])

        print(subgraphs[cluster].nodes(data=True))

    # organization='org-FKQBIvqIr7JF5Jhysdnrxx5z',
    api_key = 'sk-proj-vm0poBE6tqDcfG1OwGHhT3BlbkFJLiYmkIoWVrb2mj6RhaH1'
    openai.api_key = api_key

    # for each summary and cluster pairs, sample abstracts from the cluster and outside the cluster.
    # then ask GPT which abstracts are more similar to the summary.

    data = pd.read_csv(f"data/graphs/{name}_papers.csv")
    evaluations = {}
    for cluster, summary in summaries.items():
        # get the subgraph.
        subgraph = subgraphs[cluster]

        # get the abstracts from the cluster.
        cluster_abstracts = [data['abstract'][node] for node in subgraph.nodes() if type(data['abstract'][node]) == str]

        # get the abstracts from outside the cluster.
        outside_abstracts = [data['abstract'][node] for node in G.nodes() if node not in subgraph.nodes() and
                             type(data['abstract'][node]) == str]

        # sample 20% of the cluster size (k) and k abstracts from outside the cluster.
        cluster_abstracts = random.sample(cluster_abstracts, int(0.2 * len(cluster_abstracts)))
        outside_abstracts = random.sample(outside_abstracts, len(cluster_abstracts))

        # ask GPT which abstracts are more similar to the summary.
        cluster_avg_score = 0
        outside_avg_score = 0
        for i in range(len(cluster_abstracts)):
            # ask GPT which abstract is more similar to the summary.
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=f"How consistent is the following summary with the abstract from 1 to 100?\n"
                       f"Summary: {summary}\n"
                       f"Abstract: {cluster_abstracts[i]}\n",
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5
            )

            # get the score.
            score = response['choices'][0]['score']

            cluster_avg_score += score

            # ask GPT which abstract is more similar to the summary.
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=f"How consistent is the following summary with the abstract from 1 to 100?\n"
                       f"Summary: {summary}\n"
                       f"Abstract: {outside_abstracts[i]}\n",
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5
            )

            # get the score.
            score = response['choices'][0]['score']

            outside_avg_score += score

        # get the average scores.
        cluster_avg_score /= len(cluster_abstracts)
        outside_avg_score /= len(cluster_abstracts)

        decision = "consistent" if cluster_avg_score >= outside_avg_score else "inconsistent"

        evaluations[cluster] = (cluster_avg_score, outside_avg_score, decision)

        print(f"Cluster summary for {cluster} is {decision} with the cluster abstracts. (score: {cluster_avg_score})")

    df = pd.DataFrame(evaluations, index=['cluster_score', 'outside_score', 'decision'])
    df.to_csv(f"Evaluations/{name}_evaluations.csv")
    print(f"Finished evaluating the summaries for {name}.")
    return df
