import functions
import warnings
import summarize
import evaluate
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plot
import starry_graph as sg
import numpy as np
import networkx as nx
import pandas as pd

warnings.filterwarnings("ignore")


def run_graph_part(_name: str, _graph_kwargs: dict, _clustering_kwargs: dict, _draw_kwargs: dict,
                   _print_info: bool = False, _vertices=None, _edges=None, _distance_matrix=None):
    """
    Run the pipeline for the given name, graph_kwargs, kwargs, and draw_kwargs.
    :param _edges:  the edges of the graph.
    :param _vertices:  the vertices of the graph.
    :param _distance_matrix: the distance matrix for the embeddings.
    :param _print_info: whether to print the outputs.
    :param _name: the name of the embeddings file.
    :param _graph_kwargs: the parameters for the graph.
    :param _clustering_kwargs: the parameters for the clustering.
    :param _draw_kwargs: the parameters for the drawing.
    :return:
    """

    # create the graph.
    _G = functions.make_graph(_vertices, _edges, _distance_matrix, **_graph_kwargs)
    if _print_info:
        print(f"Graph created for '{_name}': {_G}")  # print the graph info.

    # cluster the graph.
    clusters, _G = functions.cluster_graph(_G, _name, **_clustering_kwargs)
    """
    # draw the graph.
    if wikipedia:
        # functions.draw_wiki_graph(_G, _name, **_draw_kwargs)
        functions.draw_wiki_graph(_G, _name, **_draw_kwargs)
    else:
        functions.draw_graph(_G, _name, **_draw_kwargs)
    """

    # print the results.
    if _print_info:
        print(f"{_draw_kwargs['method'].capitalize()} got {len(clusters)} "
              f"clusters for '{_name}' graph, with resolution coefficient of "
              f"{_clustering_kwargs['resolution']}.\n"
              f"Drew {int(_draw_kwargs['shown_percentage'] * 100)}% of the original graph.\n")

    return _G


def create_graph(_name, _graph_kwargs_: dict, _clustering_kwargs_: dict, _draw_kwargs_: dict, _vertices, _edges,
                 _distance_matrix=None):
    """
    Create the graph.
    :param _name:  the name of the graph.
    :param _edges:  the edges of the graph.
    :param _vertices:  the vertices of the graph.
    :param _distance_matrix: the distance matrix for the embeddings.
    :param _graph_kwargs_: the parameters for the graph.
    :param _clustering_kwargs_: the parameters for the clustering.
    :param _draw_kwargs_: the parameters for the drawing.
    :return:
    """

    _G = run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, False,
                        _vertices, _edges, _distance_matrix)

    return _G


def run_summarization(_name: str, _vertices, aspects, _print_info) -> object:
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :param _vertices: the vertices of the graph.
    :param aspects: the aspects to focus on.
    :param _print_info: whether to print the outputs.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G, _print_info)
    # summarize each cluster.
    titles = summarize.summarize_per_color(_subgraphs, _name, _vertices, aspects, _print_info)
    return titles


def plot_bar(name: str, metrics_dict: dict, vertices: pd.DataFrame = None):
    """
    Create and save a bar plot for the given metrics of a specific name and version.

    :param name: the name of the dataset.
    :param metrics_dict: Dictionary containing metrics the dataset.
    :param vertices: the vertices of the graph. (needed if purity is a valid score)
    :return:
    """
    # Retrieve metrics for the specific name and version
    values = [
        metrics_dict['avg_index'],
        metrics_dict['largest_cluster_percentage'],
        metrics_dict['purity_score'],
        metrics_dict['avg_relevancy'],
        metrics_dict['avg_coherence'],
        metrics_dict['avg_consistency'],
        metrics_dict['avg_fluency'],
        metrics_dict['success_rates'],
    ]

    # Define the labels for the x-axis
    x_labels = [
        "Average\nIndex",
        "Largest\nCluster\nPercentage",
        "Purity",
        "Average\nRelevancy",
        "Average\nCoherence",
        "Average\nConsistency",
        "Average\nFluency",
        "Success\nRate",
    ]

    # Define colors for each bar
    colors = [
        'red',  # Average Index
        'red',  # Largest Cluster Percentage
        'red',  # Purity (cluster analysis)
        'blue',  # Average Relevancy
        'blue',  # Average Coherence
        'blue',  # Average Consistency
        'blue',  # Average Fluency
        'orange',  # Success Rate
    ]

    scores = dict(zip(x_labels, zip(values, colors)))
    x_labels = []
    values = []
    colors = []
    for k, v in scores.items():
        print(f"{k}: {v}")
        if v[0] is not None:
            x_labels.append(k)
            values.append(v[0])
            colors.append(v[1])
            if k == "Purity":
                perm_vertices = vertices.copy()
                perm_vertices['label'] = np.random.permutation(perm_vertices['label']).tolist()
                perm_purity = compute_purity_score(name, perm_vertices)
                values.append(perm_purity)
                x_labels.append("Random\nPurity")
                colors.append('#FAA0A0')

    if values[0] is None:  # If the avg_index is None, remove it from the plot
        values = values[1:]
        x_labels = x_labels[1:]
        colors = colors[1:]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_labels, values, color=colors, edgecolor='black')

    # Set y-axis limits and labels
    plt.ylim(0, 1.1)
    plt.xlabel("Evaluation Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(f"Results for '{name}' Graph", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the value above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.2f}",
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    # Create custom legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Cluster Analysis'),
        Patch(facecolor='blue', edgecolor='black', label='Text Analysis'),
        Patch(facecolor='orange', edgecolor='black', label='Connection to Origin')
    ]

    # Position the legend on the left
    plt.legend(handles=legend_elements, fontsize=12, loc='upper left', bbox_to_anchor=(0, 1))

    # Adjust layout to make room for the legend on the left
    plt.tight_layout()  # Adjust the left margin

    # Define the folder path and file path
    full_path = f"Results/plots/"

    # Create the plots folder if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    full_path += f"{name}.png"

    # Save the plot
    try:
        plt.savefig(full_path)
        print(f"Plot saved at: {full_path}")
    except Exception as e:
        print(f"Error saving the plot: {e}")

    plt.show()  # Show the plot after saving
    plt.close()  # Clear the figure to free memory


def compute_silhouette_score(_name: str, _vertices, _distance_matrix, **kwargs):
    """
    Compute the silhouette score for the given name and distance matrix. The score is given by the formula:
    silhouette_score = (b - a) / max(a, b)

    where:
    a = mean intra-cluster distance
    b = mean nearest-cluster distance
    :param _name: the name of the dataset.
    :param _vertices: the vertices of the graph.
    :param _distance_matrix: the distance matrix for the embeddings.
    :param kwargs: additional parameters.
    :return:
    """

    # import the sklearn silhouette score.
    from sklearn.metrics import silhouette_score

    print_info = kwargs.get('print_info', False)  # Whether to print the results

    # Load the graph
    G = functions.load_graph(_name)

    # assign labels to the vertices based on the colors
    labels = [G.nodes[v]['color'] for v in G.nodes]
    color_to_int_map = {color: i for i, color in enumerate(set(labels))}
    # make sure the smallest is 0
    labels = [color_to_int_map[color] for color in labels]
    dists = _distance_matrix  # Load the distance matrix

    # Compute the silhouette score
    np.fill_diagonal(dists, 0)  # Set the diagonal to 0
    score = silhouette_score(dists, labels, metric='precomputed')

    if print_info:
        print(f"Silhouette coefficient: {score:.4f}")

    return score


def compute_purity_score(_name: str, _vertices, **kwargs):
    """
    Compute the purity score for the given clustered graph:
    purity = (1 / N) * sum(max(Pi))

    where:
    N = total number of vertices
    Pi = number of vertices in cluster i that belong to the majority class of cluster i
    :param _name: the name of the dataset.
    :param _vertices: the vertices of the graph.
    :param kwargs:  additional parameters.
    :return:
    """
    print_info = kwargs.get('print_info', False)  # Whether to print the results

    # Load the graph
    G = functions.load_graph(_name)

    # Check if there is a class label for the vertices, and if not, return None
    if 'label' not in _vertices.columns:
        return None

    # separate the graph into clusters.
    clusters = set(nx.get_node_attributes(G, 'color').values())
    clusters = {cluster: [node for node, color in nx.get_node_attributes(G, 'color').items() if color == cluster]
                for cluster in clusters}

    vertex_to_cluster = {}
    for i, (cluster, nodes) in enumerate(clusters.items()):
        for node in nodes:
            vertex_to_cluster[node] = i

    verts_with_predictions = _vertices.copy()  # Copy the vertices dataframe.

    # Add the predictions to the vertices dataframe.
    verts_with_predictions['prediction'] = [vertex_to_cluster[node] for node in verts_with_predictions['id']]

    # save the vertices with predictions to a csv file.
    # verts_with_predictions.to_csv('data/newsgroups_with_predictions.csv', index=False)

    # print in order.
    total = 0
    for i, cluster in enumerate(clusters):
        # get the vertices in the cluster.
        vertices_in_cluster = verts_with_predictions[verts_with_predictions['prediction'] == i]
        # get the most common label in the cluster.
        # get the values in the label column.
        labels = vertices_in_cluster['label']
        # it there is only one label per vertex, the mode will return that label.
        # check that all the labels are integer.
        if all(isinstance(label, int) for label in labels):
            # turn the labels into a numpy array.
            labels = np.array(labels)
            # get the most common label.
            most_common_label = np.bincount(labels).argmax()
        # else - multiple labels per vertex, so we need to get the most common label.
        else:
            # for each vertex, add a count to every label it has.
            label_counts = {}
            for label in labels:
                # make a list of the integers in the label.
                # remove '[' and ']' from the label.
                # split by ', ' and convert to int.
                label = label.replace('[', '').replace(']', '').split(', ')
                label = [int(val) for val in label]
                for val in label:
                    if val not in label_counts:
                        label_counts[val] = 0
                    label_counts[val] += 1
            # get the most common label.
            most_common_label = label_counts[list(label_counts.keys())[0]]
            for k, v in label_counts.items():
                if v > most_common_label:
                    most_common_label = v

            print(f"Most common label: {most_common_label}, "
                  f"cluster size: {len(vertices_in_cluster)}")

            total += most_common_label
            continue

        # get the purity score. (the number of vertices with the most common label divided by the number
        # of vertices in the cluster)
        n_most_common = len(vertices_in_cluster[vertices_in_cluster['label'] == most_common_label])
        total += n_most_common
        purity = n_most_common / len(vertices_in_cluster)
        if print_info:
            print(f'Cluster {cluster}: Purity: {purity:.2f} ({len(vertices_in_cluster)} vertices, {n_most_common}'
                  f' with the common label), Most common label: {most_common_label}')

    total_purity = total / len(verts_with_predictions)
    print(f'Total purity: {total_purity:.2f}')

    return total_purity


def compute_vmeasure_score(_name: str, _vertices, **kwargs):
    """
    Compute the V-measure score for the given clustered graph:
    v_measure = (2 * homogeneity * completeness) / (homogeneity + completeness)

    where:
    homogeneity = 1 - H(C|K) / H(C)
    completeness = 1 - H(K|C) / H(K)
    :param _name: the name of the dataset.
    :param _vertices:  the vertices of the graph.
    :param kwargs:  additional parameters.
    :return:
    """

    print_info = kwargs.get('print_info', False)  # Whether to print the results

    # use the pre-built method from scikit-learn
    from sklearn.metrics.cluster import v_measure_score

    # Load the graph
    G = functions.load_graph(_name)

    # Divide the graph into clusters
    clusters = {}
    for node in G.nodes(data=True):
        if node[1]['color'] not in clusters:
            clusters[node[1]['color']] = []
        clusters[node[1]['color']].append(node[0])

    # Check if there is a class label for the vertices, and if not, return None
    if 'label' not in _vertices.columns:
        return None

    # assign predicted labels to clusters.
    predicted_labels = {}
    for i, cluster in enumerate(clusters):
        for vertex in clusters[cluster]:
            predicted_labels[vertex] = i

    true_labels = []
    pred_list = []
    for vertex in G.nodes:
        true_labels.append(_vertices.loc[vertex, 'label'])
        pred_list.append(predicted_labels[vertex])

    # Compute the V-measure score
    v_measure = v_measure_score(true_labels, pred_list)

    if print_info:
        print(f"V-measure score: {v_measure:.4f}")

    return v_measure


def compute_jaccard_index(_name: str, _vertices, **kwargs):
    """
    Compute the Jaccard index for the given clustered graph:
    Jaccard index = |A ∩ B| / |A ∪ B|

    where:
    A = true labels
    B = predicted labels

    :param _name:
    :param _vertices:
    :param kwargs:
    :return:
    """

    print_info = kwargs.get('print_info', False)  # Whether to print the results

    # Load the graph
    G = functions.load_graph(_name)

    # Divide the graph into clusters
    clusters = {}
    for node in G.nodes(data=True):
        if node[1]['color'] not in clusters:
            clusters[node[1]['color']] = []
        clusters[node[1]['color']].append(node[0])

    # Check if there is a class label for the vertices, and if not, return None
    if 'label' not in _vertices.columns:
        return None

    # assign predicted labels to clusters.
    predicted_labels = {}
    for i, cluster in enumerate(clusters):
        for vertex in clusters[cluster]:
            predicted_labels[vertex] = i

    true_labels = []
    pred_list = []
    for vertex in G.nodes:
        true_labels.append(_vertices.loc[vertex, 'label'])
        pred_list.append(predicted_labels[vertex])

    # Compute the Jaccard index
    true_labels = np.array(true_labels)
    pred_list = np.array(pred_list)

    # Compute the Jaccard index
    intersection = np.sum(true_labels == pred_list)
    union = len(true_labels) + len(pred_list) - intersection
    jaccard_index = intersection / union

    if print_info:
        print(f"Jaccard index: {jaccard_index:.4f}")

    return jaccard_index


def the_almighty_function(pipeline_kwargs: dict):
    """
    Run the pipeline for the given parameters.
    The pipeline has the following steps:
    1. Create the graph. (either with embeddings or without)
    2. Cluster the graph.
    3. Summarize the clusters.
    4. Evaluate the success rate.
    5. Iteratively repeat steps 1-4.
    6. Plot the results and make the html component.

    The plot is then saved at 'Results/plots' folder, the summaries are saved at 'Results/summaries' folder,
    and the html component is saved at 'Results/html' folder.

    :param pipeline_kwargs:  The full pipeline parameters.
    :return:
    """

    # Unpack the pipeline parameters
    graph_kwargs = pipeline_kwargs.get('graph_kwargs', {
        "size": 2000,
        "K": 5,
        "color": "#1f78b4"
    })
    clustering_kwargs = pipeline_kwargs.get('clustering_kwargs', {
        "method": "louvain",
        "resolution": 0.5,
        "save": True
    })
    """draw_kwargs = pipeline_kwargs.get('draw_kwargs', {
        "save": True,
        "method": "louvain",
        "shown_percentage": 0.3
    })"""
    print_info = pipeline_kwargs.get('print_info', False)
    iteration_num = pipeline_kwargs.get('iteration_num', 1)  # default is 1
    vertices = pipeline_kwargs.get('vertices', None)
    edges = pipeline_kwargs.get('edges', None)
    name = pipeline_kwargs.get('name', "")
    distance_matrix = pipeline_kwargs.get('distance_matrix', None)
    aspects = pipeline_kwargs.get('aspects', None)  # expecting a list of aspects
    update_factor = pipeline_kwargs.get('update_factor', 0.5)  # default is 0.5

    assert vertices is not None, "Vertices must be provided."

    # Summarize each text beforehand if 'summary' is not a column in the data.
    if 'summary' not in vertices.columns:
        print('Summarizing the text...')
        vertices['summary'] = [summarize.summarize_text(s) for s in vertices['abstract']]

        # save to the vertices file.
        vertices.to_csv(pipeline_kwargs['vertices_path'], index=False)

    # Create the graph.
    G = functions.make_graph(vertices, edges, distance_matrix, **graph_kwargs)

    # Set initial values for sr.
    sr = 0

    # Iteratively repeat the followings:
    """
    1. Cluster the graph.
    2. Summarize the clusters.
    3. Evaluate the success rate.
    4. Update the edges using STAR graph.
    """
    kill_switch = False
    for i in range(iteration_num):
        if print_info:
            print(f"Starting iteration {i + 1}...")
            print(f"Clustering the graph for '{name}' using method '{clustering_kwargs["method"]}'...")
        # Cluster the graph.
        functions.cluster_graph(G, name, **clustering_kwargs)

        if print_info:
            print(50 * "-")
            print(f"Summarizing the clusters for '{name}'...")
        # Summarize the clusters.
        run_summarization(name, vertices, aspects, print_info)

        G = functions.load_graph(name)  # Load the graph (after assigning clusters according to the summaries)

        # Evaluate the success rate and create the STAR graph
        if print_info:
            print(50 * "-")
            print(f"Evaluating the success rate for '{name}'...")
        sr_new = sg.starr(name, vertices, G)

        # Check if the kill switch is activated
        if sr_new == -1:
            kill_switch = True
            break  # Should already have a fully summarized graph.

        if print_info:
            print(f"Success rate for '{name}' in iteration {i + 1}: {sr_new:.4f}")
            print(50 * "-")
            print(f"Updating the edges using STAR graph for '{name}'...")

        # Update the edges.
        G = sg.update(name, G, update_factor)
        sr = sr_new  # Update the success rate
        if print_info:
            print("Edges updated using STAR graph.")
            print(50 * "-")

    # Load the graph
    G = functions.load_graph(name)

    # Initialize the metrics
    rel, coh, con, flu = 0, 0, 0, 0  # Initialize the metrics

    # Evaluate and plot the metrics
    cluster_scores = functions.evaluate_clusters(name, distance_matrix)  # Evaluate the clusters

    # Improve the summaries iteratively
    for iter_ in range(iteration_num):

        if kill_switch:
            break  # Skip the rest of the iterations

        # compute the metrics
        rel_new, coh_new, con_new, flu_new, scores = evaluate.metrics_evaluations(name, vertices, G)
        if print_info:
            print(f"Metrics for '{name}' in iteration {iter_ + 1}:")
            print(f"Relevancy: {rel_new:.2f}, Coherence: {coh_new:.2f}, "
                  f"Consistency: {con_new:.2f}, Fluency: {flu_new:.2f}")
            print(50 * "-" if iter_ < iteration_num - 1 else "")

        rel_flag = 1 if rel_new > rel else 0
        coh_flag = 1 if coh_new > coh else 0
        con_flag = 1 if con_new > con else 0
        flu_flag = 1 if flu_new > flu else 0

        if sum([rel_flag, coh_flag, con_flag, flu_flag]) < 3:  # If less than 3 metrics have improved, break
            break

        rel, coh, con, flu = rel_new, coh_new, con_new, flu_new  # Update the metrics
        summarize.improve_summaries(name, vertices, scores)

    # Update the metrics dictionary

    # Check if the avg_index exists (i.e. we have more than one value to unpack in cluster_scores)
    if isinstance(cluster_scores, tuple):
        avg_index, largest_cluster_percentage = cluster_scores
    else:
        avg_index = None
        largest_cluster_percentage = cluster_scores

    # Update the metrics dictionary
    metrics_dict = {
        'avg_index': avg_index,
        'largest_cluster_percentage': largest_cluster_percentage,
        'avg_relevancy': rel,
        'avg_coherence': coh,
        'avg_consistency': con,
        'avg_fluency': flu,
        'success_rates': sr
    }

    # Update the JSON file with the new metrics
    known_metrics = json.load(open("metrics.json", "r"))
    known_metrics["avg_index"][name] = avg_index
    known_metrics["largest_cluster_percentage"][name] = largest_cluster_percentage
    known_metrics["avg_relevancy"][name] = rel
    known_metrics["avg_coherence"][name] = coh
    known_metrics["avg_consistency"][name] = con
    known_metrics["avg_fluency"][name] = flu
    known_metrics["success_rates"][name] = sr

    # Compute the silhouette score
    sil_score = compute_silhouette_score(name, vertices, distance_matrix, print_info=print_info)

    # Compute the purity score
    purity_score = compute_purity_score(name, vertices, print_info=print_info)
    metrics_dict['purity_score'] = purity_score

    # Compute the V-measure score
    v_measure_score = compute_vmeasure_score(name, vertices, print_info=print_info)

    # Compute the Jaccard index
    jaccard_index = compute_jaccard_index(name, vertices, print_info=print_info)

    # save the silhouette score, purity score, v_measure_score, and jaccard index to a csv file
    lst = [sil_score, purity_score, v_measure_score, jaccard_index]
    # get the number of clusters.
    n_clusters = len(set(G.nodes()[v]['color'] for v in G.nodes))
    lst.append(n_clusters)
    with open(f'{name}_metrics.csv', 'a') as f:
        # headers
        f.write("Silhouette Score, Purity Score, V-measure Score, Jaccard Index, # of clusters\n")
        f.write(f"{lst[0]}, {lst[1]}, {lst[2]}, {lst[3]}, {lst[-1]}\n")

    # Update the metrics dictionary with the new scores
    known_metrics["silhouette_score"][name] = sil_score
    known_metrics["purity_score"][name] = purity_score
    known_metrics["v_measure_score"][name] = v_measure_score
    known_metrics["jaccard_index"][name] = jaccard_index

    # Save the updated metrics to the JSON file
    with open("metrics.json", "w") as f:
        json.dump(known_metrics, f, indent=4)

    # Plot the metrics for the current dataset
    plot_bar(name, metrics_dict, vertices)

    # Create the html component.
    plot.plot(name, vertices)
