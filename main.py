import warnings

import pandas as pd
import one_to_rule_them_all

warnings.filterwarnings("ignore")
"""WIKIPEDIA = ["apple", "car", "clock", "London", "turtle"]"""
WIKIPEDIA = ["clock"]
"""['3D printing', "additive manufacturing", "autonomous drones", "composite material", "hypersonic missile", 
"nuclear reactor", "quantum computing", "scramjet", "smart material", "wind tunnel"] """
RAFAEL = ['hypersonic missile']


def load_params(config_file_path):
    import json
    with open(config_file_path, 'r') as _f:
        _params = json.load(_f)
    return _params


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
        'name': params['name'],
    }

    if params['use_embedding']:
        import pickle as pkl
        path = f'data/distances/{params['name']}_papers_embeddings.pkl'  # Path to the distances file.
        try:
            with open(path, 'rb') as f:
                distances = pkl.load(f)
            pipeline_kwargs['distance_matrix'] = distances['Distances']  # Load the distances.
        except FileNotFoundError:  # If the file is not found, the embedding will be computed.
            pipeline_kwargs['use_embedding'] = True
            """placeholder for now. Later will be replaced with code to '2_embed_abstract.py' and '3_embed_graph.py' 
            to compute both the embeddings and distances.
            """
    else:
        pipeline_kwargs['distance_matrix'] = None  # If the embedding is not used, the distance matrix is None.
    # Run the pipeline.
    one_to_rule_them_all.the_almighty_function(pipeline_kwargs)
