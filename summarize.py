"""
Yoel Ashkenazi
this file is responsible for summarizing the result graphs.
"""
import os
import pickle as pkl
from typing import List
import pandas as pd
import networkx as nx
from transformers import AutoTokenizer
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig
import torch


def load_graph(name: str, version: str, proportion, k: int = 5) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :param version: the version of the graph.
    :param k: the KNN parameter.
    :param proportion: the proportion of the graph.
    :return:
    """

    assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', " \
                                                               "or 'proportion'."

    if version == 'distances':
        graph_path = f"data/processed_graphs/k_{k}/only_distances/{name}.gpickle"

    elif version == 'original':
        graph_path = f"data/processed_graphs/k_{k}/only_original/{name}.gpickle"

    else:
        if proportion != 0.5:
            graph_path = f"data/processed_graphs/k_{k}/{name}_proportion_{proportion}.gpickle"
        else:
            graph_path = f"data/processed_graphs/k_{k}/{name}.gpickle"

    # load the graph.

    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    s = len(nodes)
    nodes = [node for node, data in nodes if not pd.isna(node)]
    print(
        f"Successfully removed {s - len(nodes)} nan {'vertex' if s - len(nodes) == 1 else 'vertices'} from the graph.")
    graph = graph.subgraph(nodes)

    return graph


def filter_by_colors(graph: nx.Graph) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to the community colors.
    :param graph: the graph.
    :return:
    """
    # first we filter articles by vertex type.
    articles = [node for node in graph.nodes() if graph.nodes.data()[node]['type'] == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    nodes = graph.nodes(data=True)
    colors = set([node[1]['color'] for node in nodes])
    subgraphs = []
    for i, color in enumerate(colors):  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]

        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph], name: str, version: str, proportion: float, save: bool = False,
                        k: int = 5):
    """
    This method summarizes each of the subgraphs' abstract texts using PRIMER, prints the results and save them
    to a .txt file.
    :param name: The name of the dataset.
    :param subgraphs: List of subgraphs.
    :param version: The version of the graph.
    :param proportion: The proportion of the graph.
    :param save: Whether to save the results.
    :param k: The KNN parameter.
    :return:
    """

    assert version in ['distances', 'original', 'proportion'], "Version must be one of 'distances', 'original', " \
                                                               "or 'proportion'."
    result_file_path = f"Summaries/k_{k}/{name}"
    if version == 'distances':
        result_file_path += '_only_distances/'
    elif version == 'original':
        result_file_path += '_only_original/'
    else:
        if proportion != 0.5:
            result_file_path += f'_proportion_{proportion}/'

    if save:
        # clear the results folder path.
        for file in os.listdir(result_file_path):
            os.remove(f"{result_file_path}/{file}")

    # define the model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('./PRIMERA_model/')
    config = LongformerEncoderDecoderConfig.from_pretrained('./PRIMERA_model/')
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
        './PRIMERA_model/', config=config)
    PAD_TOKEN_ID = tokenizer.pad_token_id
    DOCSEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("<doc-sep>")

    # load the abstracts.
    PATH = f'data/graphs/{name}_papers.csv'

    # save only the id and abstract columns.
    df = pd.read_csv(PATH)[['id', 'abstract']]

    # summarize each subgraph.
    for i, subgraph in enumerate(subgraphs):

        color = list(subgraph.nodes(data=True))[0][1]['color']

        num_nodes = len(subgraph.nodes())
        # skip clusters of 1- nothing to summarize.
        if len(subgraph.nodes()) == 1:
            continue

        # if the cluster is too large, summarize only the first 10 vertices.
        if len(subgraph.nodes()) > 10:
            subgraph = subgraph.subgraph(list(subgraph.nodes())[:10])

        # step 1- get the abstracts.
        abstracts = [abstract for id_, abstract in df.values if id_ in subgraph.nodes()]
        # clean nulls.
        abstracts = [abstract for abstract in abstracts if not pd.isna(abstract)]

        # step 2- summarize the abstracts.
        print(f"Summarizing {len(abstracts)} abstracts...")

        # encode the text.
        input_ids = []
        for abstract in abstracts:
            if len(input_ids) >= 4096:
                break
            # encode each abstract separately.
            input_ids.extend(
                tokenizer.encode(
                    abstract,
                    truncation=True,
                    max_length=4096 // len(abstracts),
                )
            )

            # add a document separator token.
            input_ids.append(DOCSEP_TOKEN_ID)

        # add start and end tokens.
        input_ids = (
            [tokenizer.bos_token_id]
            + input_ids
            + [tokenizer.eos_token_id]
        )
        # if the input is too long, truncate it.
        if len(input_ids) > 4096:
            input_ids = input_ids[:4096]

        # add padding.
        input_ids += [PAD_TOKEN_ID] * (4096 - len(input_ids))  # add padding.
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # add global attention mask.
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)

        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1

        # generate the summary.
        generated_ids = model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=1024,
            num_beams=5,
        )

        # decode the summary.
        summary = tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )[0]

        # print the summary.
        print(f"Summary: {summary}")

        # save the summary.
        if save:

            vers = 'vertices' if num_nodes == 1 else 'vertex'

            file_name = f'{result_file_path}/cluster_{i + 1}_{color}_{num_nodes}_{vers}_summary.txt'

            try:
                with open(file_name, 'w') as f:
                    f.write(summary)
            except FileNotFoundError:  # create the directory if it doesn't exist.
                os.makedirs(result_file_path)
                with open(file_name, 'w') as f:
                    f.write(summary)
