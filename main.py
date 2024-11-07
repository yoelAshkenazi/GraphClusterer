import functions
import warnings
import summarize
import evaluate
import json
import os

warnings.filterwarnings("ignore")
"""WIKIPEDIA = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]"""

"""ALL_NAMES = ['3D printing', "additive manufacturing", "composite material", "autonomous drones", "hypersonic missile",
             "nuclear reactor", "scramjet", "wind tunnel", "quantum computing", "smart material"]"""
ALL_NAMES = ["apple", "car", "clock", "house", "London", "mathematics", "snake", "turtle"]


def run_graph_part(_name: str, _graph_kwargs: dict, _clustering_kwargs: dict, _draw_kwargs: dict,
                   _print_info: bool = False):
    """
    Run the pipeline for the given name, graph_kwargs, kwargs, and draw_kwargs.
    :param _print_info: whether to print the outputs.
    :param _name: the name of the embeddings file.
    :param _graph_kwargs: the parameters for the graph.
    :param _clustering_kwargs: the parameters for the clustering.
    :param _draw_kwargs: the parameters for the drawing.
    :return:
    """
    # print description.
    print("\n" + "-" * 50 + f"\nCreating and clustering a graph for '{_name}' dataset...\n" + "-" * 50 + "\n")
    # create the graph.
    _G = functions.make_graph(_name, **_graph_kwargs)
    if _print_info:
        print(f"Graph created for '{_name}': {_G}")  # print the graph info.

    # cluster the graph.
    clusters = functions.cluster_graph(_G, _name, **_clustering_kwargs)

    # draw the graph.
    functions.draw_graph(_G, _name, **_draw_kwargs)

    # print the results.
    if _print_info:
        print(f"{_draw_kwargs['method'].capitalize()} got {len(clusters)} "
              f"clusters for '{_name}' graph, with distance threshold of "
              f"{_graph_kwargs['distance_threshold']} and resolution coefficient of "
              f"{_clustering_kwargs['resolution']}.\n"
              f"Drew {int(_draw_kwargs['shown_percentage'] * 100)}% of the original graph.\n")

    return functions.analyze_clusters(_G)


def run_summarization(_name: str) -> object:
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    titles = summarize.summarize_per_color(_subgraphs, _name) #
    return titles #


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

    


def evaluate_and_plot():
    """
    Evaluate and plot metrics for all combinations of names and versions.

    :param _proportion: the proportion of the graph.
    :param _K: the KNN parameter.
    :param _weight: the weight of the edges.
    :param optimized: whether to use the optimized version of the graph.
    :return:
    """
    avg_index = {_name: {} for _name in ALL_NAMES}  # the average index for each dataset.
    largest_cluster_percentage = {_name: {} for _name in ALL_NAMES}  # the largest cluster percentage.
    avg_relevancy = {_name: {} for _name in ALL_NAMES}  # the average relevancy for each dataset.
    avg_coherence = {_name: {} for _name in ALL_NAMES}  # the average coherence for each dataset.
    avg_consistency = {_name: {} for _name in ALL_NAMES}  # the average consistency for each dataset.
    avg_fluency = {_name: {} for _name in ALL_NAMES}  # the average fluency for each dataset.
    success_rates = {_name: {} for _name in ALL_NAMES}  # the success rates for each dataset.


    for _name in ALL_NAMES:
        G = functions.load_graph(_name)
        avg_index[_name], largest_cluster_percentage[_name] = functions.evaluate_clusters(G, _name)
        success_rates[_name] = evaluate.evaluate(_name, G)
        a, b, c, d = evaluate.metrics_evaluations(_name, G)
        avg_relevancy[_name] = a
        avg_coherence[_name] = b
        avg_consistency[_name] = c
        avg_fluency[_name] = d
    metrics_dict = {
        'avg_index': avg_index,
        'largest_cluster_percentage': largest_cluster_percentage,
        'avg_relevancy': avg_relevancy,
        'avg_coherence': avg_coherence,
        'avg_consistency': avg_consistency,
        'avg_fluency': avg_fluency,
        'success_rates': success_rates
    }

    for _name in ALL_NAMES:
        evaluate.plot_bar(_name, metrics_dict)


if __name__ == '__main__':
    # Set the parameters for the graph and summarization.
    proportion = 0.5
    weight = 10 ** (-0.7)  # the weight for the edges.
    res = 0.15
    K = 5

    print_info = True
    graph_kwargs = {
        'A': 15,
        'size': 2000,
        'color': '#1f78b4',
        'distance_threshold': 0.55,
        'proportion': proportion,
        'K': K,
        'weight': weight
    }

    # Set the parameters for the clustering.
    clustering_kwargs = {
        'save': True,
        'method': 'louvain',
        'resolution': res,
        'weight': weight,
        'proportion': proportion,
        'K': K
    }

    # Set the parameters for the drawing.
    draw_kwargs = {
        'save': True,
        'method': 'louvain',
        'shown_percentage': 0.3
    }

    """
    Note- The pipeline is divided into 3 parts: finding the distances, creating and clustering the graph, 
    and summarizing each cluster. The first part is done in '2_embed_abstract.py' and '3_calc_energy.py', 
    the second part is done in 'functions.py', and the third part is done in 'summarize.py'.
    For parts 1 and 2 you need python >=3.8, and for part 3 you need python 3.7.
    """

    # Step 1: Create the graphs for all versions. (need to do only once per choice of parameters)
    # graph_kwargs['K'] = K
    # clustering_kwargs['K'] = K
    # create_graph(graph_kwargs, clustering_kwargs, draw_kwargs)


    """
    # Step 2: Summarize the clusters
    os.makedirs("Summaries/optimized", exist_ok=True)
    titles_dict = {}
    for _name in ALL_NAMES:
        print(f"Running summarization for '{_name}'.")
        titles = run_summarization(_name)
        titles_dict[_name] = titles
    # Save titles_dict as a JSON file
    output_path = os.path.join("Summaries", "optimized", "Titles.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(titles_dict, f, ensure_ascii=False, indent=4)
    """
    # Step 3: Evaluate the results.
    # evaluate_and_plot()
