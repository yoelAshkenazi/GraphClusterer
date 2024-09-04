import os
import pandas as pd
import functions
import warnings
#import summarize
import evaluate
import random

warnings.filterwarnings("ignore")
ALL_NAMES = ['3D printing', "additive manufacturing", "composite material", "autonomous drones", "hypersonic missile",
             "nuclear reactor", "scramjet", "wind tunnel", "quantum computing", "smart material"]
VERSION = ['distances', 'original', 'proportion']

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


def run_summarization(_name: str, _version: str, _proportion: float, _save: bool = False, _k: int = 5,
                      _weight: float = 1):
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :param _version: the version of the graph.
    :param _proportion: the proportion of the graph.
    :param _save: whether to save the results.
    :param _k: the KNN parameter.
    :param _weight: the weight of the edges.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name, _version, _proportion, _k, _weight)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    summarize.summarize_per_color(_subgraphs, _name, _version, _proportion, _save, _k, _weight)


def create_graphs_all_versions(_graph_kwargs_: dict, _clustering_kwargs_: dict, _draw_kwargs_: dict,):
    """
    Create the graphs for all versions.

    :param _graph_kwargs_: the parameters for the graph.
    :param _clustering_kwargs_: the parameters for the clustering.
    :param _draw_kwargs_: the parameters for the drawing.
    :return:
    """
    """
    First run the proportion version (p=q=0.5).
    """
    distances_only = True
    original_only = True
    proportion_ = 0.5
    _graph_kwargs_['use_only_distances'] = distances_only
    _graph_kwargs_['use_only_original'] = original_only
    _graph_kwargs_['proportion'] = proportion_
    _clustering_kwargs_['use_only_distances'] = distances_only
    _clustering_kwargs_['use_only_original'] = original_only
    _clustering_kwargs_['proportion'] = proportion_

    for _name in ALL_NAMES:
        run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, print_info)

    """
    Then run the distances only version.
    """
    distances_only = True
    original_only = False
    proportion_ = 0.5
    _graph_kwargs_['use_only_distances'] = distances_only
    _graph_kwargs_['use_only_original'] = original_only
    _graph_kwargs_['proportion'] = proportion_
    _clustering_kwargs_['use_only_distances'] = distances_only
    _clustering_kwargs_['use_only_original'] = original_only
    _clustering_kwargs_['proportion'] = proportion_

    for _name in ALL_NAMES:
        run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, print_info)

    """
    Then run the original only version.
    """
    distances_only = False
    original_only = True
    proportion_ = 0.5
    _graph_kwargs_['use_only_distances'] = distances_only
    _graph_kwargs_['use_only_original'] = original_only
    _graph_kwargs_['proportion'] = proportion_
    _clustering_kwargs_['use_only_distances'] = distances_only
    _clustering_kwargs_['use_only_original'] = original_only
    _clustering_kwargs_['proportion'] = proportion_

    for _name in ALL_NAMES:
        run_graph_part(_name, _graph_kwargs_, _clustering_kwargs_, _draw_kwargs_, print_info)


if __name__ == '__main__':
    # set the parameters for the graph and summarization.
    use_only_distances = {'distances': True, 'original': False, 'proportion': True}
    use_only_original = {'distances': False, 'original': True, 'proportion': True}
    proportion = 0.5
    version = 'original'
    weight = 10 ** (-0.7)  # the weight for the edges.
    res = 0.15
    K = 5

    print_info = True
    graph_kwargs = {'A': 15, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.55,
                    'use_only_distances': use_only_distances[version], 'use_only_original': use_only_original[version],
                    'proportion': proportion, 'K': K, 'weight': weight}

    # set the parameters for the clustering.
    clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': res,
                         'use_only_distances': use_only_distances[version], 'weight': weight,
                         'use_only_original': use_only_original[version], 'proportion': proportion, 'K': K}

    # set the parameters for the drawing.
    draw_kwargs = {'save': True, 'method': 'louvain', 'shown_percentage': 0.3}

    """
    Note- The pipeline is divided into 3 parts: finding the distances, creating and clustering the graph, 
    and summarizing each cluster. The first part is done in '2_embed_abstract.py' and '3_calc_energy.py', 
    the second part is done in 'functions.py', and the third part is done in 'summarize.py'.
    For parts 1 and 2 you need python >=3.8, and for part 3 you need python 3.7.
    """

    # run the pipeline.
    """
    Step 1- create the graphs for all versions. (need to do only once per choice of parameters)
    """
    graph_kwargs['K'] = K
    clustering_kwargs['K'] = K
    # create_graphs_all_versions(graph_kwargs, clustering_kwargs, draw_kwargs)

    # sizes = {}
    avg_index = {name: {} for name in ALL_NAMES}  # the average index for each dataset.
    largest_cluster_percentage = {name: {} for name in ALL_NAMES}  # the largest cluster percentage.
    avg_relevancy = {name: {} for name in ALL_NAMES}  # the average relevancy for each dataset.
    avg_coherence = {name: {} for name in ALL_NAMES}  # the average coherence for each dataset.
    avg_consistency = {name: {} for name in ALL_NAMES}  # the average consistency for each dataset.
    avg_fluency = {name: {} for name in ALL_NAMES}  # the average fluency for each dataset.
    success_rates = {name: {} for name in ALL_NAMES}  # the success rates for each dataset.
    """
    Step 2- summarize the clusters for all versions.
    Step 3- evaluate the results.
    """
    pairs = [(random.choice(ALL_NAMES), random.choice(VERSION)) for _ in range(1)]

    for name, version in pairs:
        print(f"'{name}' with {version}, {proportion if version == 'proportion' else ''},"
                  f" {weight if weight != 1 else ''} graph.")
        G = functions.load_graph(name, version, proportion, K, weight)
        avg_index[name][version], largest_cluster_percentage[name][version] =  functions.evaluate_clusters(G, name)
        success_rates[name][version] = evaluate.evaluate(name, version, proportion, K, weight)
        avg_relevancy[name][version], avg_coherence[name][version], avg_consistency[name][version], avg_fluency[name][version] = evaluate.myEval(name, version, proportion, K, weight)
          

    metrics_dict = {
    'avg_index': avg_index,
    'largest_cluster_percentage': largest_cluster_percentage,
    'avg_relevancy': avg_relevancy,
    'avg_coherence': avg_coherence,
    'avg_consistency': avg_consistency,
    'avg_fluency': avg_fluency,
    'success_rates': success_rates
    }

    for (name, version) in pairs:
        evaluate.plot_bar(name, version, metrics_dict,proportion, K, weight)