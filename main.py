import warnings

import pandas as pd
import pickle as pkl

import one_to_rule_them_all

warnings.filterwarnings("ignore")
WIKIPEDIA = ["apple", "car", "clock", "London", "turtle"]
REFAEL = ['3D printing', "additive manufacturing", "autonomous drones", "composite material", "hypersonic missile",
          "nuclear reactor", "quantum computing", "scramjet", "smart material", "wind tunnel"]
ALL_NAMES = WIKIPEDIA + REFAEL


def load_params(config_file_path):
    import json
    with open(config_file_path, 'r') as _f:
        _params = json.load(_f)
    return _params


def get_distance_matrix(path_):
    if path_ == "":  # If the path is empty, return None.
        return None
    with open(path_, 'rb') as f:  # Load the distances.
        distances_ = pkl.load(f)
    return distances_  # Return the distances.


if __name__ == '__main__':
    params = load_params('config.json')

    # Set the parameters for the pipeline.
    pipeline_kwargs = {
        'graph_kwargs': params['graph_kwargs'],
        'clustering_kwargs': params['clustering_kwargs'],
        'draw_kwargs': params['draw_kwargs'],
        'print_info': params['print_info'],
        'iteration_num': params['iteration_num'],
        'vertices': pd.read_csv(params['vertices_path']),
        'edges': pd.read_csv(params['edges_path']),
        'distance_matrix': get_distance_matrix(params['distance_matrix_path']),
        'name': params['name'],
    }

    if params["allow_user_prompt"]:  # If the user prompt is allowed.
        user_aspects = input("Enter the aspects you want to focus on, separated by commas: ").split(",")
        pipeline_kwargs['aspects'] = user_aspects

    # Run the pipeline.
    one_to_rule_them_all.the_almighty_function(pipeline_kwargs)
