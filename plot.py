import os
import re  # Import the regex module
import webbrowser
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

def plot(name, wikipedia=False):
    """
    Plot the graph with the given name.
    :param name: the name of the graph.
    :param wikipedia: whether the graph is from wikipedia.
    :return: None
    """
    # -----------------------------
    # 1. Load Cluster Titles CSV
    # -----------------------------
    titles_csv_path = f"Results/starry/{name}_titles.csv"
    if not os.path.exists(titles_csv_path):
        print(f"Error: CSV file '{titles_csv_path}' does not exist.")
        return

    titles_data = pd.read_csv(titles_csv_path)
    # Ensure columns are correctly named
    expected_columns = {'Letter', 'Cluster Title'}
    if not expected_columns.issubset(titles_data.columns):
        print(f"Error: CSV file must contain columns: {expected_columns}")
        return

    # -----------------------------
    # 2. Load Main Data CSV
    # -----------------------------
    main_csv_path = f"Results/starry/{name}.csv"
    if not os.path.exists(main_csv_path):
        print(f"Error: CSV file '{main_csv_path}' does not exist.")
        return

    data = pd.read_csv(main_csv_path)

    # -----------------------------
    # 3. Ensure 'id' Contains Valid URLs
    # -----------------------------
    base_url = "https://www.example.com/"  # Modify as needed
    data['id'] = data['id'].apply(
        lambda x: x if str(x).startswith(('http://', 'https://')) else f"{base_url}{x}"
    )

    # Create a mapping from index to id (URLs)
    index_to_id = dict(zip(data['index'], data['id']))

    # -----------------------------
    # 4. Initialize NetworkX Graph
    # -----------------------------
    G = nx.Graph()

    # -----------------------------
    # 5. Iterate Through Cluster Titles to Build the Graph
    # -----------------------------
    summaries = {}  # Dictionary to hold summaries for each cluster
    for idx, row in titles_data.iterrows():
        letter = row['Letter']
        cluster_title = row['Cluster Title']

        # Define the path to the summary text file
        if wikipedia:
            summary_txt_path = os.path.join("Results", "Summaries", "Wikipedia", name, f"{cluster_title}.txt")
        else:
            summary_txt_path = os.path.join("Results", "Summaries", "Rafael", name, f"{cluster_title}.txt")

        # Read the summary from the text file
        try:
            with open(summary_txt_path, 'r', encoding='utf-8') as f:
                summary = f.read().strip()
        except FileNotFoundError:
            summary = "Summary not available."
            print(f"Warning: Summary file not found for cluster '{cluster_title}' at '{summary_txt_path}'.")
        except Exception as e:
            summary = "Error loading summary."
            print(f"Error reading summary for cluster '{cluster_title}': {e}")

        # Replace periods with line breaks intelligently
        # This regex replaces a period with a newline only if it's followed by a space and a capital letter or end of string
        formatted_summary = re.sub(r'\.(?=\s+[A-Z]|$)', '.\n', summary)

        # Store the formatted summary in the summaries dictionary
        summaries[letter] = formatted_summary

        # Combine cluster title and summary for the tooltip
        combined_title = f"{cluster_title}\n{formatted_summary}"

        # Add the central node with Cluster Title as label
        G.add_node(
            letter,
            label=cluster_title,
            title=combined_title,
            url=formatted_summary,  # Store the formatted summary directly
            color='#ffcc00',
            size=40,
            font_size=20,  # Custom attribute for font size
            is_central=True  # Custom attribute to identify central nodes
        )

        # -----------------------------
        # 6. Add Peripheral Nodes and Edges
        # -----------------------------
        # Filter rows with the current title
        cluster_data = data[data['title'] == cluster_title]
        if wikipedia:
            PATH = f'data/wikipedia/{name}_100_samples_nodes.csv'
        else:
            PATH = f'data/graphs/{name}_papers.csv'

        DATA = pd.read_csv(PATH)[['id', 'abstract']]

        peripheral_nodes = []

        for _, peripheral_row in cluster_data.iterrows():
            index = peripheral_row['index']
            total_in_score = peripheral_row['total_in_score']
            total_out_score = peripheral_row['total_out_score']

            # Ensure index exists in index_to_id
            if index not in index_to_id:
                print(f"Warning: Index {index} not found in 'id' mapping.")
                continue

            peripheral_id = index_to_id[index]
            # Fetch the abstract
            try:
                peripheral_abstract = data.loc[data['id'] == peripheral_id, 'abstract'].iloc[0]
            except IndexError:
                peripheral_abstract = "Abstract not available."
                print(f"Warning: Abstract not found for ID '{peripheral_id}'.")

            # Format the abstract by replacing '.' with '.\n' intelligently
            formatted_abstract = re.sub(r'\.(?=\s+[A-Z]|$)', '.\n', peripheral_abstract)

            # Add the peripheral node
            G.add_node(
                index,
                label=str(index),
                title=f"Abstract:\n{formatted_abstract}",
                url=formatted_abstract,  # Store the formatted abstract
                color='#1f78b4',
                size=30,
                font_size=12,
                is_central=False  # Peripheral node
            )

            # Determine edge color based on score comparison
            edge_color = 'blue' if total_in_score >= total_out_score else 'red'

            # Add the edge between central node and peripheral node
            G.add_edge(letter, index, color=edge_color, width=3)

            # Add to peripheral nodes list
            peripheral_nodes.append(index)

        # -----------------------------
        # 7. Assign Peripheral Nodes to Central Node
        # -----------------------------
        G.nodes[letter]['peripheral_nodes'] = peripheral_nodes

    # -----------------------------
    # 8. Generate Positions for Nodes Using Spring Layout
    # -----------------------------
    pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility

    # -----------------------------
    # 9. Create Edge Traces (One Scatter Trace per Edge)
    # -----------------------------
    edge_traces = []
    for edge in G.edges(data=True):
        start_node, end_node, attrs = edge
        x0, y0 = pos[start_node]
        x1, y1 = pos[end_node]
        edge_color = attrs.get('color', 'black')
        edge_width = attrs.get('width', 2)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],  # 'None' to separate edges
                y=[y0, y1, None],
                mode='lines',
                line=dict(color=edge_color, width=edge_width),
                hoverinfo='none',
                showlegend=False  # Hide individual edge traces from legend
            )
        )

    # -----------------------------
    # 10. Create Node Traces
    # -----------------------------
    node_x = []
    node_y = []
    node_text = []
    node_urls = []
    node_is_central = []
    node_ids = []
    node_summaries = []  # To hold summaries for central nodes and abstracts for peripheral nodes

    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(attrs['label'])  # Display Cluster Title or Index
        node_urls.append(attrs['url'])
        node_is_central.append(attrs.get('is_central', False))
        node_ids.append(node)
        node_summaries.append(attrs['url'])  # Store summary or abstract

    # Assign colors and sizes
    node_colors = [attrs['color'] for _, attrs in G.nodes(data=True)]
    node_sizes = [attrs['size'] for _, attrs in G.nodes(data=True)]

    # Create node trace with labels and different symbols
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',  # Include text labels
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            symbol=['star' if is_central else 'circle' for is_central in node_is_central],
            line=dict(width=2, color='#333'),
            sizemode='diameter'
        ),
        customdata=list(zip(node_urls, node_is_central, node_ids, node_summaries)),
        name='Nodes'
    )

    # -----------------------------
    # 11. Create the Plotly Figure
    # -----------------------------
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=f"Interactive Network Graph: {name}",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=150, t=40),  # Increased right margin for legend
                        annotations=[dict(
                            x=1.05,
                            y=1,
                            xref='paper',
                            yref='paper',
                            text='Edge Colors:<br>Red: High Out Score<br>Blue: High In Score',
                            showarrow=False,
                            align='left',
                            font=dict(
                                size=12
                            )
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # -----------------------------
    # 12. Convert the Figure to HTML
    # -----------------------------
    html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')

    # -----------------------------
    # 13. Define the Custom HTML and JavaScript for the Popup
    # -----------------------------
    custom_html = """
    <style>
    /* The Modal (background) */
    .modal {
        display: none; /* Hidden by default */
        position: fixed; /* Stay in place */
        z-index: 1000; /* Sit on top */
        padding-top: 50px; /* Location of the box */
        left: 0;
        top: 0;
        width: 100%; /* Full width */
        height: 100%; /* Full height */
        overflow: auto; /* Enable scroll if needed */
        background-color: rgba(0, 0, 0, 0.9); /* Black w/ opacity */
    }

    /* Modal Content */
    .modal-content {
        background-color: #000; /* Black background */
        margin: auto;
        padding: 20px;
        border: 1px solid #888;
        color: #fff; /* White text */
        white-space: pre-wrap; /* Preserve whitespace and line breaks */
        font-size: 16px; /* Increased font size for better readability */
        max-width: 90%; /* Maximum width relative to viewport */
        max-height: 80vh; /* Maximum height relative to viewport */
        overflow: auto; /* Enable scrolling if content exceeds max-height */
        box-sizing: border-box; /* Include padding and border in element's total width and height */
    }

    /* The Close Button */
    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }

    .close:hover,
    .close:focus {
        color: #fff;
        text-decoration: none;
        cursor: pointer;
    }
    </style>

    <div id="summaryModal" class="modal">
      <div class="modal-content">
        <span class="close">&times;</span>
        <p id="summaryText"></p>
      </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var modal = document.getElementById("summaryModal");
        var span = document.getElementsByClassName("close")[0];
        var summaryText = document.getElementById("summaryText");

        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_click', function(data) {
            var point = data.points[0];
            var customdata = point.customdata;
            var text = customdata[3]; // Summary or abstract is the 4th element

            if (text && text !== "") {
                // Display the text in the modal
                summaryText.textContent = text;
                modal.style.display = "block";
            }
        });

        // When the user clicks on <span> (x), close the modal
        span.onclick = function() {
            modal.style.display = "none";
        }

        // When the user clicks anywhere outside of the modal, close it
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    });
    </script>
    """

    # -----------------------------
    # 14. Inject the Custom HTML into the Plotly HTML
    # -----------------------------
    modified_html = html_str.replace("</body>", f"{custom_html}\n</body>")

    # -----------------------------
    # 15. Save the Modified HTML to a File
    # -----------------------------
    output_html_path = os.path.join("Results", "html", f"{name}.html")
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(modified_html)
        print(f"Interactive network graph saved to '{output_html_path}'.")
    except Exception as e:
        print(f"Error saving the interactive network graph HTML: {e}")
        return

    # -----------------------------
    # 16. Open the HTML File in the Default Web Browser
    # -----------------------------
    abs_path = os.path.abspath(output_html_path)
    file_url = f'file://{abs_path}'
    webbrowser.open(file_url)
