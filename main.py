import warnings

import pandas as pd
import pickle as pkl
import sys
import os
import json

import one_to_rule_them_all  # Change to relative import if needed.
import calc_energy  # Change to relative import if needed.

warnings.filterwarnings("ignore")
WIKIPEDIA = ["apple", "car", "clock", "London", "turtle"]
REFAEL = ['3D printing', "additive manufacturing", "autonomous drones", "composite material", "hypersonic missile",
          "nuclear reactor", "quantum computing", "scramjet", "smart material", "wind tunnel"]
ALL_NAMES = WIKIPEDIA + REFAEL


def load_params(config_file_path=''):
    # If the path is wrong, configure a default config file.
    if not os.path.exists(config_file_path):
        print(f"Config file not found at {config_file_path}. Making a default config file.")
        config_file_path = 'config.json'
        default_config_content = {
            "graph_kwargs": {
                "size": 2000,
                "K": 5,
                "color": "#1f78b4"
            },

            "clustering_kwargs": {
                "method": "louvain",
                "resolution": 0.5,
                "save": True
            },

            "draw_kwargs": {
                "save": True,
                "method": "louvain",
                "shown_percentage": 0.3
            },

            "name": "3D printing",  # Using 3d printing dataset by default.
            "vertices_path": "data/graphs/3D printing_papers.csv",  # vertices.
            "edges_path": "data/graphs/3D printing_graph.csv",  # edges.
            "distance_matrix_path": "data/distances/3D printing_energy_distance_matrix.pkl",  # distances.

            "iteration_num": 2,
            "print_info": True,
            "allow_user_prompt": True

        }

        with open(config_file_path, 'w') as _f:
            json.dump(default_config_content, _f)

        return default_config_content  # Return the default config file.

    with open(config_file_path, 'r') as _f:
        _params = json.load(_f)
    return _params


def get_distance_matrix(path_, name_):
    if path_ == "" or not os.path.exists(path_):  # If the path is empty or wrong, create a new distance matrix.
        print("No distance matrix provided. Creating a distance matrix...")
        dists = calc_energy.compute_energy_distance_matrix(name_, 0.9, 5.0, 5)
        # Save the distance matrix to path.
        dir_name = 'data/distances'
        output_name = name_ + '_energy_distance_matrix.pkl'
        os.makedirs(dir_name, exist_ok=True)
        with open(dir_name + '/' + output_name, 'wb') as f:
            pkl.dump(dists, f)
        return dists
    with open(path_, 'rb') as f:  # Load the distances.
        distances_ = pkl.load(f)
    return distances_  # Return the distances.


if __name__ == '__main__':

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'

    params = load_params(config_path)

    # Set the parameters for the pipeline.
    pipeline_kwargs = {
        'graph_kwargs': params['graph_kwargs'],
        'clustering_kwargs': params['clustering_kwargs'],
        'draw_kwargs': params['draw_kwargs'],
        'print_info': params['print_info'],
        'iteration_num': params['iteration_num'],
        'vertices': pd.read_csv(params['vertices_path']),
        'edges': pd.read_csv(params['edges_path']) if params['edges_path'] != "" else None,
        'distance_matrix': get_distance_matrix(params['distance_matrix_path'], params['name']),
        'name': params['name'],
    }

    if params["allow_user_prompt"]:  # If the user prompt is allowed.
        user_aspects = input("Enter the aspects you want to focus on, separated by commas: ").split(",")
        pipeline_kwargs['aspects'] = user_aspects

    # Run the pipeline.
    print("Starting the pipeline...\n\n\n")
    one_to_rule_them_all.the_almighty_function(pipeline_kwargs)
