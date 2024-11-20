import networkx as nx
import pandas as pd
import os
from typing import Dict
import cohere
from dotenv import load_dotenv
import pickle as pk
import random
import matplotlib.pyplot as plt

wikipedia = False
load_dotenv()
cohere_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client with your API key
co = cohere.Client(api_key=cohere_key)


def extract_colors(graph: nx.Graph) -> Dict[str, str]:
    """
    Extract the colors of the clusters from the graph.
    :param graph: the graph.
    :return:
    """
    title_to_color = {}
    for node in graph.nodes:
        title = graph.nodes[node].get('title', None)
        if title is not None:
            title_to_color[title] = graph.nodes[node]['color']
    return title_to_color


def starr(name: str, G: nx.Graph = None) -> float:
    """
    Load the cluster summary for the given name.
    :param name: the name of the dataset.
    :param G: the graph.
    :return:
    """
    map = {}
    global wikipedia
    if wikipedia:
        summary_path = f"Results/Summaries/wikipedia/"
        for file in os.listdir('Results/Summaries/wikipedia'):
            if file.startswith(name):
                summary_path += file
    else:
        summary_path = f"Results/Summaries/Rafael/"
        for file in os.listdir('Results/Summaries/Rafael'):
            if file.startswith(name):
                summary_path += file

    # Check if the summary directory exists and list its contents
    if not os.path.exists(summary_path):
        print(f"Error: Directory {summary_path} does not exist!")
        return None

    clusters = os.listdir(summary_path)
    summaries = {}
    titles = [cluster.split('.')[0] for cluster in clusters]  # Get the titles.
    title_to_color = extract_colors(G)
    subgraphs = {}
    for i, title in enumerate(titles):
        decode_break = False
        summary_file_path = os.path.join(summary_path, clusters[i])
        with open(summary_file_path, 'r') as f:
            try:
                summaries[title] = f.read()
            except UnicodeDecodeError:
                decode_break = True
        if decode_break:
            print(f"Failed to decode summary file: {summary_file_path}")
            continue
        # Get the subgraph.
        color = title_to_color[title]
        nodes = [node for node in G.nodes if G.nodes()[node].get('color', 'green') == color]
        subgraphs[title] = G.subgraph(nodes).copy()  # Ensure mutable subgraph

    # For each summary and cluster pairs, sample abstracts from the cluster and outside the cluster.
    if wikipedia:
        PATH = f'data/wikipedia/{name}_100_samples_nodes.csv'
    else:
        PATH = f'data/graphs/{name}_papers.csv'

    # Ensure the CSV file exists
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"The file {PATH} does not exist.")

    # Read only the id and abstract columns
    data = pd.read_csv(PATH)[['id', 'abstract', 'title']]
    id_to_title_map = {row['id']: row['title'] for _, row in data.iterrows()}
    evaluations = {}
    counter = 1
    total_in_score = 0  # Total scores for the abstracts sampled inside the clusters.
    total_out_score = 0  # Total scores for the abstracts sampled outside the clusters.

    for title, subgraph in subgraphs.items():
        # Get the subgraph.
        summary = summaries[title]
        cluster_name = title
        # Get the abstracts from the cluster.
        cluster_abstracts = {row['id']: row['abstract'] for id, row in data.iterrows() if row['id'] in subgraph.nodes}
        # Clean NaNs.
        cluster_abstracts = {id: abstract for id, abstract in cluster_abstracts.items() if not pd.isna(abstract)}

        # Get the abstracts from outside the cluster.
        outside_abstracts = [row['abstract'] for _, row in data.iterrows() if row['id'] not in subgraph.nodes]

        # Clean NaNs.
        outside_abstracts = [abstract for abstract in outside_abstracts if not pd.isna(abstract)]
        # Ask Cohere's API which abstracts are more similar to the summary.
        for id, abstract in cluster_abstracts.items():
            outside = random.choice(outside_abstracts)
            # Evaluate consistency for abstracts inside the cluster.
            prompt_in = (
                f"Answer using only a number between 1 to 100: "
                f"How consistent is the following summary with the abstract?\n"
                f"Summary: {summary}\n"
                f"Abstract: {abstract}\n"
                f"Even if the summary is not consistent with the abstract, please provide a score between 0 to 100, and only the score,"
                f" and only the score, without any '.' or ',' etc."
            )

            response_in = co.generate(
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
            prompt_out = (
                f"Answer using only a number between 0 to 100: "
                f"How consistent is the following summary with the abstract?\n"
                f"Summary: {summary}\n"
                f"Abstract: {outside}\n"
                f"Even if the summary is not consistent with the abstract, please provide a score between 0 to 100, and only the score,"
                f" and only the score, without any '.' or ',' etc."
            )

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
            title = id_to_title_map[id]

            map[counter] = {
                'index': counter,
                'total_in_score': score_in,
                'total_out_score': score_out,
                'id': id,
                'title': cluster_name
            }
            counter += 1

        decision = "consistent" if total_in_score >= total_out_score else "inconsistent"

        evaluations[cluster_name] = (total_in_score, total_out_score, decision)

        print(f"Cluster summary for cluster '{cluster_name}' is {decision} with the cluster abstracts. "
              f"\nScore in: {total_in_score}\nScore out: {total_out_score}\n{'-' * 50}")

    # Save the map to the directory 'Results/starry/{name}.csv'
    output_dir = 'Results/starry'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{name}.csv')

    # Convert the map to a DataFrame and save it as a CSV file
    map_df = pd.DataFrame.from_dict(map, orient='index')
    map_df.to_csv(output_path, index=False)
    return total_in_score / (total_in_score + total_out_score) if (total_in_score + total_out_score) != 0 else 0


def plot(name):
    # Load the CSV file
    file_path = f"Results/starry/{name}.csv"
    data = pd.read_csv(file_path)

    # Ensure output directory exists
    output_dir = "Results/starry"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the graph
    G = nx.Graph()

    # Initialize variables for labeling and legend
    title_to_letter = {}
    letter = 'a'

    # Positioning variables
    current_center_x = 0
    center_spacing = 2  # Space between star graph centers (smaller spacing)

    # Iterate through unique titles to build the graph
    for cluster_title in data['title'].unique():
        # Assign a letter to the cluster title
        title_to_letter[letter] = cluster_title

        # Filter rows with the current title
        cluster_data = data[data['title'] == cluster_title]

        # Add the central node with the letter as the label
        G.add_node(letter)

        for i, row in cluster_data.iterrows():
            index = row['index']
            total_in_score = row['total_in_score']
            total_out_score = row['total_out_score']

            # Add the node for the index
            G.add_node(index)

            # Determine edge color based on score comparison
            edge_color = 'blue' if total_in_score >= total_out_score else 'red'

            # Add the edge between the cluster center and the index node
            G.add_edge(letter, index, color=edge_color, weight=0.1)

        # Move to the next center position
        current_center_x += center_spacing

        # Increment the letter
        letter = chr(ord(letter) + 1)

    # Draw the graph
    plt.figure(figsize=(10, 8))  # Reduced figure size
    colors = [G[u][v]['color'] for u, v in G.edges()]  # Get edge colors
    pos = nx.spring_layout(G)  # Use spring layout for the graph
    nx.draw(G, pos=pos, with_labels=True, edge_color=colors,
            node_color='lightblue', node_size=500, font_size=8)

    # Save the plot
    output_path = os.path.join(output_dir, f"{name}.png")
    plt.title(f"Combined Star Graphs for {name}")
    plt.savefig(output_path, format='png')
    plt.close()

    # Save the legend as a CSV file
    legend_path = os.path.join(output_dir, f"{name}_titles.csv")
    legend_df = pd.DataFrame(list(title_to_letter.items()), columns=["Letter", "Cluster Title"])
    legend_df.to_csv(legend_path, index=False)

    print(f"Saved legend (title mappings) to {legend_path}")


def update(name, G):
    """
    Update the graph based on the 'Results/starry/{name}.csv' evaluations.
    :param name: the name of the dataset.
    :param G: the graph to be updated.
    """
    # Load the CSV file created in the 'plot' function
    csv_path = f"Results/starry/{name}.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return

    data = pd.read_csv(csv_path)

    # Initialize a list to store nodes that need to be created
    nodes_to_create = []
    edges_to_add = []

    # Iterate over unique clusters (titles)
    for cluster_title in data['title'].unique():
        # Filter data for the current cluster
        cluster_data = data[data['title'] == cluster_title]

        # Extract node indices and scores
        node_indices = cluster_data['index'].tolist()
        total_in_scores = cluster_data['total_in_score'].tolist()
        total_out_scores = cluster_data['total_out_score'].tolist()

        # Create a mapping of node -> color (blue or red)
        node_colors = {}
        for i, node in enumerate(node_indices):
            if total_in_scores[i] > total_out_scores[i]:
                node_colors[node] = 'blue'
            else:
                node_colors[node] = 'red'

        # Iterate over unique pairs of nodes (u, v) in the cluster
        n = len(node_indices)
        for i in range(n):
            for j in range(i + 1, n):  # Ensure u != v and only consider each pair once
                u = node_indices[i]
                v = node_indices[j]

                # Step (b): Add edges based on node colors
                if node_colors[u] == 'blue' and node_colors[v] == 'blue':
                    # Both u and v are blue
                    if G.has_edge(u, v):
                        current_weight = G[u][v].get('weight', 1)  # Default weight to 1 if missing
                        new_weight = current_weight * 2
                        G[u][v]['weight'] = new_weight
                        print(f"Edge ({u}, {v}) weight increased from {current_weight} to {new_weight}.")
                    else:
                        edges_to_add.append((u, v, {'weight': 1}))  # Track edge to be added
                else:
                    # At least one of u or v is red
                    if G.has_edge(u, v):
                        current_weight = G[u][v].get('weight', 1)  # Default weight to 1 if missing
                        new_weight = current_weight / 2
                        G[u][v]['weight'] = new_weight
                    else:
                        # No action needed if the edge does not exist
                        print(f"Edge ({u}, {v}) does not exist. No action taken.")

                # Keep track of nodes that need to be created
                if u not in G.nodes:
                    nodes_to_create.append(u)
                if v not in G.nodes:
                    nodes_to_create.append(v)

    # Add all nodes to the graph (if any were missing)
    for node in set(nodes_to_create):  # Use `set` to ensure unique nodes
        G.add_node(node)
        print(f"Node '{node}' added to the graph.")

    # Add all edges to the graph
    for u, v, attributes in edges_to_add:
        G.add_edge(u, v, **attributes)

    # Save the updated graph
    updated_graph_path = f"data/optimized_graphs/{name}.gpickle"
    with open(updated_graph_path, 'wb') as f:
        pk.dump(G, f)
    print(f"Updated graph saved to {updated_graph_path}")


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return: the graph.
    """
    graph_path = None
    for file in os.listdir('data/optimized_graphs'):
        if file.startswith(name):
            graph_path = os.path.join('data/optimized_graphs', file)
            break

    if graph_path is None:
        raise FileNotFoundError(f"No graph file starts with '{name}' in 'data/optimized_graphs/' directory.")

    print(name, graph_path)
    # Load the graph.
    with open(graph_path, 'rb') as f:
        graph = pk.load(f)

    # Filter the graph in order to remove the nan nodes.
    nodes = list(graph.nodes(data=True))
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes).copy()  # Ensure mutable subgraph

    return graph


def iteraroe(name):
    """
    Complete pipeline for processing each dataset.
    :param name: the name of the dataset.
    """
    G = load_graph(name)
    sr = starr(name, G)
    # plot(name)
    update(name, G)
