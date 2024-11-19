import functions
import warnings
import summarize
import evaluate
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

ALL_NAMES = None
print_info = None
wikipedia = False
warnings.filterwarnings("ignore")

def update_wikipedia():
    global wikipedia
    functions.update_wikipedia()
    summarize.update_wikipedia()
    evaluate.update_wikipedia()
    wikipedia = True

def run_graph_part(_name: str, _graph_kwargs: dict, _clustering_kwargs: dict, _draw_kwargs: dict, _print_info: bool = False):
    """
    Run the pipeline for the given name, graph_kwargs, kwargs, and draw_kwargs.
    :param _print_info: whether to print the outputs.
    :param _name: the name of the embeddings file.
    :param _graph_kwargs: the parameters for the graph.
    :param _clustering_kwargs: the parameters for the clustering.
    :param _draw_kwargs: the parameters for the drawing.
    :return:
    """
    global wikipedia
    print(f"wikipedia: {wikipedia}")
    # print description.
    print("\n" + "-" * 50 + f"\nCreating and clustering a graph for '{_name}' dataset...\n" + "-" * 50 + "\n")
    # create the graph.
    if wikipedia:
        _G = functions.make_wiki_graph(_name, **_graph_kwargs)
    else:
        _G = functions.make_graph(_name, **_graph_kwargs)
    if _print_info:
        print(f"Graph created for '{_name}': {_G}")  # print the graph info.

    # cluster the graph.
    clusters = functions.cluster_graph(_G, _name, **_clustering_kwargs)

    # draw the graph.
    if wikipedia:
        # functions.draw_wiki_graph(_G, _name, **_draw_kwargs)
        functions.draw_wiki_graph(_G, _name, **_draw_kwargs)
    else:
        functions.draw_graph(_G, _name, **_draw_kwargs)

    # print the results.
    if _print_info:
        print(f"{_draw_kwargs['method'].capitalize()} got {len(clusters)} "
              f"clusters for '{_name}' graph, with distance threshold of "
              f"{_graph_kwargs['distance_threshold']} and resolution coefficient of "
              f"{_clustering_kwargs['resolution']}.\n"
              f"Drew {int(_draw_kwargs['shown_percentage'] * 100)}% of the original graph.\n")

    return functions.analyze_clusters(_G)


def create_graph(_graph_kwargs_: dict, _clustering_kwargs_: dict, _draw_kwargs_: dict,):
    """
    Create the graph.
    :param _graph_kwargs_: the parameters for the graph.
    :param _clustering_kwargs_: the parameters for the clustering.
    :param _draw_kwargs_: the parameters for the drawing.
    :return:
    """

    proportion_ = 0.5
    _graph_kwargs_['proportion'] = proportion_
    _clustering_kwargs_['proportion'] = proportion_

    for _name in ALL_NAMES:
        run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, print_info)


def run_summarization(_name: str) -> object:
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :return:
    """
    print(_name)
    # load the graph.
    _G = summarize.load_graph(_name)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    titles = summarize.summarize_per_color(_subgraphs, _name) #
    return titles #


def evaluate_and_plot():
    """
    Evaluate and plot metrics for all combinations of names and versions.
    Additionally, maintain a JSON file that stores the metrics, updating existing entries
    or adding new ones as necessary.

    :param json_file_path: Path to the JSON file storing metrics.
    :return: None
    """
    global wikipedia
    global ALL_NAMES
    json_file_path="metrics.json"
    # Initialize metrics dictionaries
    metrics_dict = {
        'avg_index': {},
        'largest_cluster_percentage': {},
        'avg_relevancy': {},
        'avg_coherence': {},
        'avg_consistency': {},
        'avg_fluency': {},
        'success_rates': {}
    }

    # Load existing metrics from JSON if the file exists
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                existing_metrics = json.load(f)
            print(f"Loaded existing metrics from {json_file_path}.")
        except json.JSONDecodeError:
            print(f"JSON file {json_file_path} is corrupted. Starting with empty metrics.")
            existing_metrics = {}
    else:
        print(f"No existing JSON file found at {json_file_path}. Starting with empty metrics.")
        existing_metrics = {}

    # Iterate over all datasets and evaluate metrics
    for _name in ALL_NAMES:
        G = functions.load_graph(_name)
        if wikipedia:
            avg_idx, largest_cluster_pct = functions.evaluate_wiki_clusters(G)
        else:
            avg_idx, largest_cluster_pct = functions.evaluate_clusters(G,_name)
        print(f"Average Index for '{_name}': {avg_idx}, Largest Cluster Percentage: {largest_cluster_pct}") ###

        success_rate = evaluate.evaluate(_name, G) ###
        print(f"Success rate for '{_name}': {success_rate}")
        
        relevancy, coherence, consistency, fluency = evaluate.metrics_evaluations(_name, G)

        # Update metrics_dict with current evaluations
        metrics_dict['avg_index'][_name] = avg_idx
        metrics_dict['largest_cluster_percentage'][_name] = largest_cluster_pct
        metrics_dict['avg_relevancy'][_name] = relevancy
        metrics_dict['avg_coherence'][_name] = coherence
        metrics_dict['avg_consistency'][_name] = consistency
        metrics_dict['avg_fluency'][_name] = fluency
        metrics_dict['success_rates'][_name] = success_rate

        # Update existing_metrics with new or updated values
        for metric in metrics_dict:
            if metric not in existing_metrics:
                existing_metrics[metric] = {}
            existing_metrics[metric][_name] = metrics_dict[metric][_name]

    # Save the updated metrics back to the JSON file
    try:
        with open(json_file_path, 'w') as f:
            json.dump(existing_metrics, f, indent=4)
        print(f"Metrics successfully saved to {json_file_path}.")
    except Exception as e:
        print(f"Failed to save metrics to {json_file_path}: {e}")

    # Plot the metrics for each dataset
    for _name in ALL_NAMES:
        plot_bar(_name, existing_metrics)


def plot_bar(name: str, metrics_dict: dict):
    """
    Create and save a bar plot for the given metrics of a specific name and version.

    :param name: the name of the dataset.
    :param metrics_dict: Dictionary containing metrics for each (name, version) combination.
    :return:
    """
    global wikipedia
    # Retrieve metrics for the specific name and version
    values = [
        metrics_dict['avg_index'].get(name, 0),  
        metrics_dict['largest_cluster_percentage'].get(name, 0),  
        metrics_dict['avg_relevancy'].get(name, 0),  
        metrics_dict['avg_coherence'].get(name, 0),  
        metrics_dict['avg_consistency'].get(name, 0),  
        metrics_dict['avg_fluency'].get(name, 0),  
        metrics_dict['success_rates'].get(name, 0)  
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
    if wikipedia:
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

    # Sanitize the file name to avoid issues with spaces or special characters
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "_", name)

    # Define the folder path and file path
    plot_folder = "Results/plots"  # Always save in this folder
    plot_file_name = f"{sanitized_name}.png"  # Save the figure with the given name
    full_path = os.path.join(plot_folder, plot_file_name)

    # Create the plots folder if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)

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
    :param kwargs: the parameters for the pipeline.
    :return:
    """
    global ALL_NAMES, print_info
    ALL_NAMES = pipeline_kwargs['ALL_NAMES']
    print_info = pipeline_kwargs['print_info']
    graph_kwargs = pipeline_kwargs['graph_kwargs']
    clustering_kwargs = pipeline_kwargs['clustering_kwargs']
    draw_kwargs = pipeline_kwargs['draw_kwargs']
    wikipedia = pipeline_kwargs['wikipedia']
    if wikipedia:
        update_wikipedia()

    """
    Note- The pipeline is divided into 3 parts: finding the distances, creating and clustering the graph, 
    and summarizing each cluster. The first part is done in '2_embed_abstract.py' and '3_calc_energy.py', 
    the second part is done in 'functions.py', and the third part is done in 'summarize.py'.
    For parts 1 and 2 you need python >=3.8, and for part 3 you need python 3.7.
    """

    # Step 1: Create the graphs for all versions. (need to do only once per choice of parameters)
    # create_graph(graph_kwargs, clustering_kwargs, draw_kwargs)

    # Step 2: Summarize the clusters
    if wikipedia:
        os.makedirs("Results/Summaries/wikipedia", exist_ok=True)
    else:
        os.makedirs("Results/Summaries/Rafael", exist_ok=True)
    titles_dict = {}
    for _name in ALL_NAMES:
        print(f"Running summarization for '{_name}'.")
        titles = run_summarization(_name)
        titles_dict[_name] = titles
    # Save titles_dict as a JSON file
    if wikipedia:
        output_path = os.path.join("Results", "Summaries", "wikipedia", "Titles.json")
    else:
        output_path = os.path.join("Results", "Summaries", "Rafael", "Titles.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(titles_dict, f, ensure_ascii=False, indent=4)

    # Step 3: Evaluate the results.
    # evaluate_and_plot()

    