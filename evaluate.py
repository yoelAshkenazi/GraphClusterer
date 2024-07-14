"""
Yoel Ashkenazi
This file is responsible for evaluating the summarization results using GPT-4 api.
"""
import os
import pandas as pd
from openai import OpenAI
import random
import pickle as pkl

# organization='org-FKQBIvqIr7JF5Jhysdnrxx5z',
api_key = 'sk-proj-2gkj2jzXE7fAgLzqIn9TT3BlbkFJzQwHpetTBkmbSO6Q6KBl'
organization = 'org-FKQBIvqIr7JF5Jhysdnrxx5z'
# set the openai api key. and the organization. also as an environment variable.
os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(api_key=api_key, organization=organization)


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

    # for each summary and cluster pairs, sample abstracts from the cluster and outside the cluster.
    # then ask GPT which abstracts are more similar to the summary.

    data = pd.read_csv(f"data/graphs/{name}_papers.csv")[['id', 'abstract']]  # load the data.
    evaluations = {}
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

        # sample 20% of the cluster size (k) and k abstracts from outside the cluster.
        cluster_abstracts = random.sample(cluster_abstracts, int(0.2 * len(cluster_abstracts)))
        outside_abstracts = random.sample(outside_abstracts, len(cluster_abstracts))

        # ask GPT which abstracts are more similar to the summary.
        cluster_avg_score = 0
        outside_avg_score = 0
        for i in range(len(cluster_abstracts)):
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
            cluster_avg_score += score_in

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

            outside_avg_score += score_out

        # get the average scores.
        cluster_avg_score /= len(cluster_abstracts)
        outside_avg_score /= len(cluster_abstracts)

        decision = "consistent" if cluster_avg_score >= outside_avg_score else "inconsistent"

        evaluations[cluster_name] = (cluster_avg_score, outside_avg_score, decision)

        print(f"Cluster summary for cluster '{cluster_name}' is {decision} with the cluster abstracts. "
              f"\nScore in: {cluster_avg_score}\nScore out: {outside_avg_score}\n{'-' * 50}")

    df = pd.DataFrame(evaluations, index=['cluster_score', 'outside_score', 'decision'])
    try:
        df.to_csv(f"Evaluations/{name}_evaluations.csv")
    except OSError:
        os.makedirs("Evaluations/")
        df.to_csv(f"Evaluations/{name}_evaluations.csv")
    print(f"Finished evaluating the summaries for {name}.")
    return df
