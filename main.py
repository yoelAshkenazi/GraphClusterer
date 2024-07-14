import os

import functions
import warnings
# import summarize
import evaluate

warnings.filterwarnings("ignore")


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


def run_summarization(_name: str, _version: str, _proportion: float, _save: bool = False):
    """
    Run the summarization for the given name and summarize_kwargs.
    :param _name: the name of the dataset.
    :param _version: the version of the graph.
    :param _proportion: the proportion of the graph.
    :param _save: whether to save the results.
    :return:
    """
    # load the graph.
    _G = summarize.load_graph(_name)
    # filter the graph by colors.
    _subgraphs = summarize.filter_by_colors(_G)
    # summarize each cluster.
    summarize.summarize_per_color(_subgraphs, _name, _version, _proportion, _save)


if __name__ == '__main__':
    # set the parameters for the graph and summarization.
    use_only_distances = False
    use_only_original = True
    proportion = 0.5
    version = 'original'

    print_info = True
    graph_kwargs = {'A': 15, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.55,
                    'use_only_distances': use_only_distances, 'use_only_original': use_only_original,
                    'proportion': proportion}

    # set the parameters for the clustering.
    clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': 0.15,
                         'use_only_distances': use_only_distances, 'use_only_original': use_only_original,
                         'proportion': proportion}

    # set the parameters for the drawing.
    draw_kwargs = {'save': True, 'method': 'louvain', 'shown_percentage': 0.3}

    """
    Note- The pipeline is divided into 3 parts: finding the distances, creating and clustering the graph, 
    and summarizing each cluster. The first part is done in '2_embed_abstract.py' and '3_calc_energy.py', 
    the second part is done in 'functions.py', and the third part is done in 'summarize.py'.
    For parts 1 and 2 you need python >=3.8, and for part 3 you need python 3.7.
    """

    # run the pipeline.
    names = ['3D printing', 'smart material', 'additive manufacturing', 'autonomous drones', 'composite material',
             'hypersonic missile', 'nuclear reactor', 'quantum computing', 'scramjet', 'wind tunnel']

    # sizes = {}
    # in_scores = {}
    # out_scores = {}

    # for name in names:  # run the pipeline for each name with only the original distances.
    #     run_graph_part(name, graph_kwargs, clustering_kwargs, draw_kwargs, print_info)
    #     run_summarization(name, version, proportion, _save=True)
    #     a, b = evaluate.evaluate(name, version, proportion)
    #     in_scores[name] = a
    #     out_scores[name] = b
    #
    # graph_kwargs['use_only_distances'] = True
    # clustering_kwargs['use_only_distances'] = True
    # graph_kwargs['use_only_original'] = False
    # clustering_kwargs['use_only_original'] = False
    # version = 'distances'
    #
    # for name in names:  # run the pipeline for each name with only the distances.
    #     run_graph_part(name, graph_kwargs, clustering_kwargs, draw_kwargs, print_info)
    #     run_summarization(name, version, proportion, _save=True)
    #     a, b = evaluate.evaluate(name, version, proportion)
    #     in_scores[name] = a
    #     out_scores[name] = b

    # use both.
    # graph_kwargs['use_only_original'] = True
    # clustering_kwargs['use_only_original'] = True
    # version = 'proportion'

    # for name in names:  # run the pipeline for each name with both the original and distances.
    #     run_graph_part(name, graph_kwargs, clustering_kwargs, draw_kwargs, print_info)
    #     run_summarization(name, version, proportion, _save=True)
    #     a, b = evaluate.evaluate(name, version, proportion)
    #     in_scores[name] = a
    #     out_scores[name] = b

    # print the results.
    # print(f"\nIn scores: {in_scores}\nOut scores: {out_scores}")
    # print(f"\nAverage in score: {sum(in_scores.values()) / len(in_scores)}")
    # print(f"Average out score: {sum(out_scores.values()) / len(out_scores)}")
