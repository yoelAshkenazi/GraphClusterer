import networkx as nx
import pandas as pd
import functions
import warnings
import pickle
import summarize

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
    print("\n" + "-" * 50 + f"\nCreating and clustering a graph for '{_name}' dataset...\n" + "-" * 50 + "\n)")
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


if __name__ == '__main__':
    # set the parameters for the graph.
    print_info = True
    graph_kwargs = {'A': 10, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.55}

    # set the parameters for the clustering.
    clustering_kwargs = {'save': True, 'method': 'louvain', 'resolution': 0.2}

    # set the parameters for the drawing.
    draw_kwargs = {'save': True, 'method': 'louvain', 'shown_percentage': 0.3}

    """
    Note- The pipeline is divided into 3 parts: finding the distances, creating and clustering the graph, 
    and summarizing each cluster. The first part is done in '2_embed_abstract.py' and '3_calc_energy.py', 
    the second part is done in 'functions.py', and the third part is done in 'summarize.py'.
    For parts 1 and 2 you need python >=3.8, and for part 3 you need python 3.7.
    """

    # set the parameters for the summarization.
    summarize_kwargs = {'save': True, 'add_title': True}

    # run the pipeline.
    names = ['3D printing', 'additive manufacturing', 'autonomous drones',
             'composite material', 'hypersonic missile', 'nuclear reactor',
             'quantum computing', 'scramjet', 'smart material', 'wind tunnel']

    # for name in names:
    #     run_graph_part(name, graph_kwargs, clustering_kwargs, draw_kwargs, print_info)

    G = summarize.load_graph('3D printing')
    subgraphs = summarize.filter_by_colors(G)
    summarize.summarize_per_color(subgraphs, '3D printing', **summarize_kwargs)

