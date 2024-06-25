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


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return:
    """
    file_path = f"data/processed_graphs/{name}.pkl"
    with open(file_path, 'rb') as f:
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
    for color in colors:  # filter by colors.
        nodes = [node for node in graph.nodes() if graph.nodes.data()[node]['color'] == color]
        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph], name: str, **kwargs):
    """
    This method summarizes each of the subgraphs' abstract texts using PRIMER, prints the results and save them
    to a .txt file.
    :param name: The name of the dataset.
    :param subgraphs: List of subgraphs.
    :param kwargs: Additional parameters.
    :return:
    """

    # define parameters.
    save = kwargs['save'] if 'save' in kwargs else False
    add_title = kwargs['add_title'] if 'add_title' in kwargs else False

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

        # add title.
        if add_title:
            print(f"Cluster {i + 1}: no title yet.")

        # print the summary.
        print(f"Summary: {summary}")

        # save the summary.
        if save:
            # reminder to add the title part later.
            vers = 'vertices' if num_nodes > 1 else 'vertex'
            file_name = f'Summaries/{name}/cluster_{i + 1}_{num_nodes}_{vers}_summary.txt'

            try:
                with open(file_name, 'w') as f:
                    f.write(summary)
            except FileNotFoundError:  # create the directory if it doesn't exist.
                os.makedirs(f'Summaries/{name}')
                with open(file_name, 'w') as f:
                    f.write(summary)
