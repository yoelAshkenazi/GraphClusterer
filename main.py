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
    G = functions.make_graph(name, **graph_kwargs)
    if _print_info:
        print(f"Graph created for '{name}': {G}")  # print the graph info.

    # cluster the graph.
    clusters = functions.cluster_graph(G, **clustering_kwargs)

    # draw the graph.
    functions.draw_graph(G, name, **draw_kwargs)

    # print the results.
    if _print_info:
        print(f"{draw_kwargs['method'].capitalize()} got {len(clusters)} "
              f"clusters for '{name}' graph, with distance threshold of "
              f"{graph_kwargs['distance_threshold']}.")


if __name__ == '__main__':
    name = '3D printing'  # the name of the embeddings file.
    # set the parameters for the graph.
    print_info = False
    graph_kwargs = {'A': 10, 'size': 2000, 'color': '#1f78b4', 'distance_threshold': 0.5}
    # set the parameters for the clustering.
    clustering_kwargs = {'method': 'louvain', 'resolution': 1.5}
    # set the parameters for the drawing.
    draw_kwargs = {'save': True, 'method': 'louvain', 'shown_percentage': 0.5}

    # run the pipeline.
    names = ['3D printing', 'additive manufacturing', 'autonomous drones',
             'composite material', 'hypersonic missile', 'nuclear reactor',
             'quantum computing', 'scramjet', 'smart material', 'wind tunnel']
    for name in names:
        run(name, graph_kwargs, clustering_kwargs, draw_kwargs)
