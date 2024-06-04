import functions
import warnings

warnings.filterwarnings("ignore")


def run(_name: str, _graph_kwargs: dict, _clustering_kwargs: dict, _draw_kwargs: dict,
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
    # create the graph.
    G = functions.make_graph(_name, **_graph_kwargs)
    if _print_info:
        print(f"Graph created for '{_name}': {G}")  # print the graph info.

    # cluster the graph.
    clusters = functions.cluster_graph(G, **_clustering_kwargs)

    # draw the graph.
    functions.draw_graph(G, _name, **_draw_kwargs)

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
    graph_kwargs = {'save': True, 'A': 10, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.6}
    # set the parameters for the clustering.
    clustering_kwargs = {'method': 'louvain', 'resolution': 0.15}
    # set the parameters for the drawing.
    draw_kwargs = {'save': True, 'method': 'louvain', 'shown_percentage': 0.3}

    # run the pipeline.
    names = ['3D printing', 'additive manufacturing', 'autonomous drones',
             'composite material', 'hypersonic missile', 'nuclear reactor',
             'quantum computing', 'scramjet', 'smart material', 'wind tunnel']
    for name in names:
        run(name, graph_kwargs, clustering_kwargs, draw_kwargs, print_info)
