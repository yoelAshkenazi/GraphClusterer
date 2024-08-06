import os
import pandas as pd
import functions
import warnings
import summarize

# import evaluate

warnings.filterwarnings("ignore")
ALL_NAMES = ['3D printing', "additive manufacturing", "composite material", "autonomous drones", "hypersonic missile",
             "nuclear reactor", "scramjet", "wind tunnel", "quantum computing", "smart material"]


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


def run_summarization(_name: str, _version: str, _proportion: float, _save: bool = False, _k: int = 5):
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :param _version: the version of the graph.
    :param _proportion: the proportion of the graph.
    :param _save: whether to save the results.
    :param _k: the KNN parameter.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name, _version, _proportion)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    summarize.summarize_per_color(_subgraphs, _name, _version, _proportion, _save, _k)


def create_graphs_all_versions(_graph_kwargs_: dict, _clustering_kwargs_: dict, _draw_kwargs_: dict, ):
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
    K = 10

    print_info = True
    graph_kwargs = {'A': 15, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.55,
                    'use_only_distances': use_only_distances[version], 'use_only_original': use_only_original[version],
                    'proportion': proportion, 'K': K}

    # set the parameters for the clustering.
    clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': 0.15,
                         'use_only_distances': use_only_distances[version],
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
    in_scores = {name: {} for name in ALL_NAMES}
    out_scores = {name: {} for name in ALL_NAMES}
    success_rates = {name: {} for name in ALL_NAMES}  # the success rates for each dataset.
    """
    Step 2- summarize the clusters for all versions.
    Step 3- evaluate the results.
    """
    for name in ALL_NAMES[8:]:  # run the pipeline for each name with only the original distances.
        for version in ['distances', 'original', 'proportion']:
            print(f"'{name}' with {version} graph.")
    #         # a, b, _ = evaluate.evaluate(name, version, proportion, K)
    #         # in_scores[name][version] = a
    #         # out_scores[name][version] = b
    #         # success_rates[name][version] = a / (a + b) if a + b != 0 else 0
    #         # print(f"Success rate for '{name}' with {version} graph: {success_rates[name][version]}")
            run_summarization(name, version, proportion, _save=True, _k=K)

"""
Step 4- save the results.
"""
# save the results
# df1 = pd.DataFrame(in_scores)
# df2 = pd.DataFrame(out_scores)
# df3 = pd.DataFrame(success_rates)
#
# # save the results.
# try:
#     df1.to_csv(f'Results/k_{K}/in_scores.csv')
#     df2.to_csv(f'Results/k_{K}/out_scores.csv')
#     df3.to_csv(f'Results/k_{K}/success_rates.csv')
# except OSError:
#     os.makedirs('Results', exist_ok=True)
#     os.makedirs(f'Results/k_{K}', exist_ok=True)
#     df1.to_csv(f'Results/k_{K}/in_scores.csv')
#     df2.to_csv(f'Results/k_{K}/out_scores.csv')
#     df3.to_csv(f'Results/k_{K}/success_rates.csv')
#
# # print the results.
# print(f"\nIn scores: {in_scores}\nOut scores: {out_scores}")
# print(f"\nAverage in score: {sum(in_scores.values()) / len(in_scores)}")
# print(f"Average out score: {sum(out_scores.values()) / len(out_scores)}")
