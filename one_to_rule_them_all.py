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

    proportion_ = 0.5
    _graph_kwargs_['proportion'] = proportion_
    _clustering_kwargs_['proportion'] = proportion_

    _G = run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, False,
                        _vertices, _edges, _distance_matrix)

    return _G


def run_summarization(_name: str, _vertices, aspects) -> object:
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :param _vertices: the vertices of the graph.
    :param aspects: the aspects to focus on.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    titles = summarize.summarize_per_color(_subgraphs, _name, _vertices, aspects)
    return titles


def plot_bar(name: str, metrics_dict: dict):
    """
    Create and save a bar plot for the given metrics of a specific name and version.

    :param name: the name of the dataset.
    :param metrics_dict: Dictionary containing metrics the dataset.
    :return:
    """
    # Retrieve metrics for the specific name and version
    values = [
        metrics_dict['avg_index'],
        metrics_dict['largest_cluster_percentage'],
        metrics_dict['avg_relevancy'],
        metrics_dict['avg_coherence'],
        metrics_dict['avg_consistency'],
        metrics_dict['avg_fluency'],
        metrics_dict['success_rates']
    ]

    # Define the labels for the x-axis
    x_labels = [
        "Average\nIndex",
        "Largest\nCluster\nPercentage",
        "Average\nRelevancy",
        "Average\nCoherence",
        "Average\nConsistency",
        "Average\nFluency",
        "Success\nRate"
    ]

    # Define colors for each bar
    colors = [
        'red',    # Average Index
        'red',    # Largest Cluster Percentage
        'blue',   # Average Relevancy
        'blue',   # Average Coherence
        'blue',   # Average Consistency
        'blue',   # Average Fluency
        'orange'  # Success Rate
    ]
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
    draw_kwargs = pipeline_kwargs.get('draw_kwargs', {
        "save": True,
        "method": "louvain",
        "shown_percentage": 0.3
    })
    print_info = pipeline_kwargs.get('print_info', False)
    iteration_num = pipeline_kwargs.get('iteration_num', 1)
    vertices = pipeline_kwargs.get('vertices', None)
    edges = pipeline_kwargs.get('edges', None)
    name = pipeline_kwargs.get('name', "")
    distance_matrix = pipeline_kwargs.get('distance_matrix', None)
    aspects = pipeline_kwargs.get('aspects', None)  # expecting a list of aspects

    assert vertices is not None, "Vertices must be provided."
    assert edges is not None, "Edges must be provided."

    # Create and cluster the graph.
    create_graph(name, graph_kwargs, clustering_kwargs, draw_kwargs, vertices, edges, distance_matrix)

    # Summarize the clusters
    titles = run_summarization(name, vertices, aspects)

    # Iteratively repeat the followings:
    """
    1. Evaluate the success rate.
    2. Create the STAR graph.
    3. Update the edges.
    4. Cluster.
    5. Summarize.
    """
    for i in range(iteration_num):

        if print_info:  # Print the titles if necessary.
            print(f"Cluster titles for '{name}' in iteration {i + 1}: {titles}")

        # Load the graph.
        G = functions.load_graph(name)

        # Evaluate the success rate and create the STAR graph
        sr = sg.starr(name, vertices, G)
        if print_info:
            print(f"Success rate for '{name}' in iteration {i+1}: {sr}")
            print(50*"-")

        # Update the edges.
        G = sg.update(name, G)

        # Cluster the graph.
        functions.cluster_graph(G, name, **clustering_kwargs)

        # Summarize the clusters.
        titles = run_summarization(name, vertices, aspects)

    """
    Placeholder for now. Will be replaced with code to iteratively improve the summary based on the 4 textual metrics.
    """
    # Load the graph
    G = functions.load_graph(name)

    # Evaluate and plot the metrics
    cluster_scores = functions.evaluate_clusters(name, distance_matrix)  # Evaluate the clusters
    success_rate = sg.starr(name, vertices, G)  # Evaluate the success rate

    # Improve the summaries iteratively
    for iter_ in range(iteration_num):
        rel, coh, con, flu, scores = evaluate.metrics_evaluations(name, vertices, G)
        if print_info:
            print(f"Metrics for '{name}' in iteration {iter_ + 1}:")
            print(f"Relevancy: {rel:.2f}, Coherence: {coh:.2f}, Consistency: {con:.2f}, Fluency: {flu:.2f}")
            print(50 * "-")
        summarize.improve_summaries(name, vertices, scores)

    # Update the metrics dictionary
    rel, coh, con, flu, _ = evaluate.metrics_evaluations(name, vertices, G)

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
        'success_rates': success_rate
    }

    # Update the JSON file with the new metrics
    known_metrics = json.load(open("metrics.json", "r"))
    known_metrics["avg_index"][name] = avg_index
    known_metrics["largest_cluster_percentage"][name] = largest_cluster_percentage
    known_metrics["avg_relevancy"][name] = rel
    known_metrics["avg_coherence"][name] = coh
    known_metrics["avg_consistency"][name] = con
    known_metrics["avg_fluency"][name] = flu
    known_metrics["success_rates"][name] = success_rate

    with open("metrics.json", "w") as f:  # Save the updated metrics
        json.dump(known_metrics, f, indent=4)

    # Plot the metrics for the current dataset
    plot_bar(name, metrics_dict)

    # Create the html component.
    plot.plot(name, vertices)
