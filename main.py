import warnings
import one_to_rule_them_all

warnings.filterwarnings("ignore")
WIKIPEDIA = ["apple", "car", "clock", "London", "turtle"]
"""'3D printing', "additive manufacturing", "autonomous drones", "composite material",
 "quantum computing", "scramjet", "smart material", "wind tunnel"
 """
RAFAEL = ["hypersonic missile", "nuclear reactor"]

if __name__ == '__main__':
    # Set the parameters for the graph and summarization.
    proportion = 0.5
    weight = 10 ** (-0.7)  # the weight for the edges.
    res = 0.15
    K = 5
    print_info = True
    wikipedia = False
    iteration_num = 2

    # Set the parameters for the graph.
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

    # Set the parameters for the pipeline.
    pipeline_kwargs = {
        'graph_kwargs': graph_kwargs,
        'clustering_kwargs': clustering_kwargs,
        'draw_kwargs': draw_kwargs,
        'print_info': print_info,
        'ALL_NAMES': WIKIPEDIA if wikipedia else RAFAEL,
        'wikipedia': wikipedia,
        'iteration_num': iteration_num
    }

    # Run the pipeline.
    one_to_rule_them_all.the_almighty_function(pipeline_kwargs)
