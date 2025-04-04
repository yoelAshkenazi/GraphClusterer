"""
Yoel Ashkenazi
this file is responsible for summarizing the result graphs.
"""
import os
import pandas as pd
from typing import List
import networkx as nx
import cohere
from dotenv import load_dotenv
import pickle as pkl
import replicate


# Integrate dotenv to load environment variables
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

def reconnect(key: str):
    """
    Reconnect to the Cohere API.
    :param key: the API key.
    """
    return cohere.Client(api_key=key)


def upload_graph(graph: nx.Graph, name: str) -> None:
    """
    Upload the graph to the given path.
    :param graph: the graph to upload.
    :param name: the name of the graph.
    :return: None
    """
    graph_path = f'data/clustered_graphs/{name}.gpickle'

    # save the graph.
    with open(graph_path, 'wb') as f:
        pkl.dump(graph, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_graph(name: str) -> nx.Graph:
    """
    Load the graph with the given name.
    :param name: the name of the graph.
    :return: the graph.
    :return:
    """
    graph_path = f'data/clustered_graphs/{name}.gpickle'  # the path to the graph.

    # load the graph.
    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)

    # filter the graph in order to remove the nan nodes.
    nodes = graph.nodes(data=True)
    nodes = [node for node, data in nodes if not pd.isna(node)]

    graph = graph.subgraph(nodes)

    return graph


def filter_by_colors(graph: nx.Graph, print_info: str) -> List[nx.Graph]:
    """
    Partition the graph into subgraphs according to the vertex colors.
    :param graph: the graph.
    :param print_info: whether to print the information.
    :return:
    """
    # Filter non-paper nodes.
    articles = [node for node in graph.nodes if graph.nodes()[node].get('type', '') == 'paper']
    articles_graph = graph.subgraph(articles)
    graph = articles_graph

    # second filter divides the graph by colors.
    colors = set([graph.nodes()[n]['color'] for n in graph.nodes])
    subgraphs = []
    for i, color in enumerate(colors):  # filter by colors.
        nodes = [node for node in graph.nodes if graph.nodes()[node]['color'] == color]
        subgraph = graph.subgraph(nodes)
        subgraphs.append(subgraph)
        if print_info:  # If to print the information.
            print(f"color {color} has {len(subgraph.nodes)} vertices.")

    return subgraphs


def summarize_per_color(subgraphs: List[nx.Graph], name: str, vertices: pd.DataFrame, aspects: List, print_info: bool) \
        -> List[str]:
    """
    Summarizes each subgraph's abstract texts using Cohere's API, prints the results, and optionally saves them to
    text files.

    :param subgraphs: List of subgraphs.
    :param name: The name of the dataset.
    :param vertices: The vertices DataFrame.
    :param aspects: The aspects to focus on.
    :param print_info: Whether to print the information.
    :return: None
    """
    if aspects is None:
        aspects = []
    result_file_path = f"Results/Summaries/{name}/"  # the path to save the results.

    # Ensure the directory exists
    os.makedirs(result_file_path, exist_ok=True)

    # Clean previous summaries by removing existing files in the directory
    if os.path.exists(result_file_path):
        for file in os.listdir(result_file_path):  # remove all the files in the directory.
            os.remove(os.path.join(result_file_path, file))

    # Integrate dotenv to load environment variables
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API_KEY")

    # api_token = os.getenv("LLAMA_API_KEY")

    # Load the abstracts from the CSV file
    df = vertices[['id', 'summary']]  # todo- check summary vs abstract.
    count_titles = 2
    titles_list = []
    vertex_to_title_map = {}  # map from vertex to title.
    # Iterate over each subgraph to generate summaries
    for i, subgraph in enumerate(subgraphs, start=1):

        co = reconnect(cohere_api_key)

        # Extract abstracts corresponding to the nodes in the subgraph
        node_ids = set(subgraph.nodes())
        abstracts = df[df['id'].isin(node_ids)]['summary'].dropna().tolist()  # todo - check summary vs abstract.
        color = subgraph.nodes[list(subgraph.nodes())[0]]['color']  # get the color of the subgraph.

        # If only one abstract is present, skip summarization.
        if len(abstracts) <= 1 and print_info:
            print(f"Cluster {color}: Insufficient abstracts, Skipping.")
            continue

        # If there are too many abstracts (over 1000), select a random subset of 1000 abstracts.
        if len(abstracts) > 1000:
            import random
            abstracts = random.sample(abstracts, 1000)

        # Combine all abstracts into a single text block with clear delimiters and instructional prompt
        combined_abstracts = " ".join([f"<New Text:> {abstract}" for j, abstract in enumerate(abstracts)])
        with open('prompt for command-r.txt', 'r') as file:
            instructions_command_r = file.read()

        if len(aspects) > 0:  # If aspects are given.
            aspects_str = ", ".join(aspects)
            aspects_instruction = "In your summary, focus on the following aspects: " + aspects_str + "."

            # Inject the aspects instruction into the prompt, in the one before the last line.
            instructions_command_r = instructions_command_r.split('\n')
            instructions_command_r[-2] = aspects_instruction
            instructions_command_r = '\n'.join(instructions_command_r)

        # make sure the combined length of the texts doesn't exceed the maximum allowed by the API.
        if len(combined_abstracts) > 100000:
            combined_abstracts = combined_abstracts[:100000]
            combined_abstracts = combined_abstracts[:combined_abstracts.rfind('<New Text:>')]  # remove the last text.

        # Generate the summary using Cohere's summarize API
        bad_response_flag = False
        try:
            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=instructions_command_r + combined_abstracts,
                max_tokens=1000,
            )
        except Exception as e:  # If an error occurred.
            bad_response_flag = True
            print(f"Error in Cohere API: {e}.\nSkipping.")
            summary = ('An error occurred while generating the summary.\n'
                       'Please check the logs for more information.')
            co = reconnect(cohere_api_key)  # Reconnect to the API.

        if not bad_response_flag:
            summary = response.generations[0].text.strip()

        # Refine the summary by generating a title using the LLAMA API
        with open('prompt for llama.txt', 'r') as file:
            instructions_llama = file.read()
        input_params = {
            "prompt": instructions_llama + summary,
            "max_tokens": 1000
        }
        output = replicate.run(
            "meta/meta-llama-3.1-405b-instruct",
            input=input_params
        )
        summary = "".join(output)

        # Generate a unique title if the title already exists
        with open('prompt for llama 2.txt', 'r') as file:
            instructions_llama2 = file.read()
        prompt = instructions_llama2 + summary
        if titles_list:  # If there are already titles.
            prompt += (f"\n\nTry to avoid giving one of those titles: {titles_list}\n"
                       f"Your suggested title must be in English!")
        input_params = {
            "prompt": prompt,
            "max_tokens": 300
        }
        try:
            output = replicate.run(
                "meta/meta-llama-3.1-405b-instruct",
                input=input_params
            )
        except Exception as e:
            print(f"Error in LLAMA API: {e}.\nSkipping.")
            output = ['Generic Title']
            continue
        title = "".join(output)
        title = title.replace('"', '')

        # make sure the title doesn't end with a newline character or a '.'.
        if title[-1] in ['\n', '.', ' ']:
            title = title[:-1]

        if print_info:  # If to print the information.
            print(f"Cluster {color}: {title} ({len(abstracts)} textual vertices)")

        if title in titles_list:
            title = f"{title} ({count_titles})"
            count_titles += 1
        titles_list.append(title)
        # save the summary.
        num_nodes = len(subgraph)
        vers = 'vertices' if num_nodes != 1 else 'vertex'

        file_name = f'{title} ({num_nodes} {vers}).txt'

        try:
            with open(result_file_path + file_name, 'w', encoding='utf-8') as f:
                f.write(summary)
                # print(f"Summary saved to {result_file_path + file_name}")
        except FileNotFoundError:  # create the directory if it doesn't exist.
            os.makedirs(result_file_path, exist_ok=True)
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(summary)
        except UnicodeEncodeError:
            # make the summary encoding compatible
            summary = summary.encode('ascii', 'ignore').decode()
            with open(result_file_path + file_name, 'w', encoding='utf-8') as f:
                f.write(summary)
                # print(f"Summary saved to {result_file_path + file_name}")

        for v in subgraph.nodes():
            vertex_to_title_map[v] = f'{title} ({num_nodes} {vers})'

    # Add the titles to the graph
    G = load_graph(name)
    nx.set_node_attributes(G, values=vertex_to_title_map, name='title')
    upload_graph(G, name)
    return titles_list


def improve_summary(summary, data, scores, score_names, name, title):
    """
    Re-create the summary with a focused prompt on the scores, so that the scores will be higher later.
    :param summary: the summary.
    :param data: the data.
    :param scores: the scores.
    :param score_names: the names of the scores.
    :param name: the name of the file.
    :param title: the title of the summary.
    :return: the improved summary.
    """

    PROMPT = """
             In the following task, you will receive a summary and the original texts that the summary was created from. 
             {TASK_DESCRIPTION}
             You will receive the summary and the texts separated by '<New Text>' between each consecutive pair of 
             texts. After the texts, you will receive the current summary.
             You will need to improve the summary according to the following Instructions:
             
             1- You must not directly reference any of the texts.
             2- Your summary must be between 5-10 sentences long.
             3- Your summary must mention key ideas and concepts that repeat in the texts.
             4- You must not invent any information. The summary must contain only information directly deduced from 
                the texts.
             5- Your summary must be coherent, fluent in language, and relevant in content to the texts.
             6- In your summary, only refer to the texts you get as input, do not make things up in the summary.
             7- Try your best that the summary will be the most relevant, coherent, consistent and as fluent as can be.
             8- Do not use characters that are outside the standard ASCII range.
             
             Texts:{TEXTS}
             
             Summary:{SUMMARY}
             """

    co = reconnect(cohere_api_key)

    # Create a list of texts to summarize.
    texts = data['abstract'].tolist()  # Get the abstracts.
    texts = [abstract for abstract in texts if pd.notna(abstract)]  # Remove NaN values.

    # if there are too many texts (over 1000), select a random subset of 1000 texts.
    if len(texts) > 1000:
        import random
        texts = random.sample(texts, 1000)

    texts = [f"<New Text:> {text}" for text in texts]  # Add the prompt to each text.
    TEXTS = " ".join(texts)  # Combine the texts.

    # Load the instructions for the llama model. (for refining)
    with open('prompt for llama.txt', 'r') as file:
        instructions_llama = file.read()

    TASK_DESCRIPTION = ""
    for score, name_ in zip(scores, score_names):  # For each score.

        if score >= 0.8:  # If the score is high enough, continue.
            continue

        if TASK_DESCRIPTION == "":
            TASK_DESCRIPTION = f"Your task is to improve the summary's {name_}.\n"

        else:
            TASK_DESCRIPTION += f'Also, you need to improve the summary\'s {name_}.\n'

    # Combine the prompt with the texts.
    prompt = PROMPT.format(TASK_DESCRIPTION=TASK_DESCRIPTION, TEXTS=TEXTS, SUMMARY=summary)

    # Generate the summary using Cohere's summarize API
    try:
        response = co.generate(
            model='command-r-plus-08-2024',
            prompt=prompt,
            max_tokens=500
        )
        summary = response.generations[0].text.strip()  # Get the summary.

    except Exception as e:
        print(f"Error in Cohere API: {e}.\nKeeping summary unchanged.")
        return

    # Refine the summary using Llama.
    input_params = {
        "prompt": instructions_llama + summary,
        "max_tokens": 500
    }

    try:
        output = replicate.run(
            "meta/meta-llama-3.1-405b-instruct",
            input=input_params,
        )
        summary = "".join(output)  # Get the summary.
    except Exception as e:
        print(f"Error in LLAMA API: {e}.\nKeeping summary unchanged.")
        return

    # Save the summary. (after improving all necessary scores)
    path = f'Results/Summaries/{name}/{title}.txt'  # the path to save the results.
    # Make sure the path leads to an empty file.
    if os.path.exists(path):
        os.remove(path)
    try:
        with open(path, 'w', encoding="utf-8") as f:
            f.write(summary)
    except UnicodeEncodeError:
        summary = summary.encode('ascii', 'ignore').decode()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(summary)
    except FileNotFoundError:
        print(f"File not found: {name} {title}.")
        exit(4)

    return summary  # Return the summary.


def improve_summaries(name, vertices, titles_to_scores):
    """
    Improve the summaries of the given data.
    :param name:  The name of the dataset.
    :param vertices: The vertices DataFrame.
    :param titles_to_scores:  The scores of the titles.
    :return:
    """
    G = load_graph(name)  # Load the graph.

    SCORE_NAMES = ['coherence', 'relevance', 'consistency', 'fluency']

    for title, scores in titles_to_scores.items():
        # Get the summary.
        with open(f"Results/Summaries/{name}/{title}.txt", 'r', encoding='utf-8') as file:
            summary = file.read()

        # Get the relevant data.
        texts = [node for node in G.nodes if 'title' in G.nodes()[node]]
        G = G.subgraph(texts)
        relevant_ids = [node for node in G.nodes() if G.nodes()[node]['title'] == title]
        data = vertices[vertices['id'].isin(relevant_ids)]

        # Improve the summary.
        improve_summary(summary, data, scores, SCORE_NAMES, name, title)


def summarize_text(s: str):
    """
    Summarize the given text.
    :param s:  the text to summarize.
    :return:
    """

    INSTRUCTIONS = ("In this task you are required to summarize the given text. \n"
                    "Your summary must be between 5-10 sentences long. \n"
                    "Your summary must be as coherent as possible, and must not use phrases"
                    "like 'in this text'.\n Your summary must not include sentences and information "
                    "that is outside the text. \nYou may use sentences from the text without citing them.")

    co = reconnect(cohere_api_key)
    if not isinstance(s, str):  # If the text is None, return None.
        return None

    l_s = len(s.split(' '))  # Get the number of words in the text.

    if l_s > 300:  # must summarize.
        flag = True
        while flag:
            try:
                response = co.generate(
                    model='command-r-plus-08-2024',
                    prompt=INSTRUCTIONS + s,
                    max_tokens=500
                )
                flag = False
            except Exception as e:
                co = reconnect(cohere_api_key)

        summary = response.generations[0].text.strip()
        return summary

    return s  # Return the text as is.
