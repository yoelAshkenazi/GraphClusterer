import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a directed graph
G = nx.DiGraph()

# Define the central node and the days of the week
G.add_node("Week")
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Add edges from the central "Week" node to each day
for day in days:
    G.add_node(day)
    G.add_edge("Week", day)

# Define positions for each day around the central "Week" node in a circular layout
pos = {"Week": (0.5, 0.5)}
angle_step = 2 * np.pi / len(days)  # Calculate the angle step to place nodes in a circular layout
radius = 0.3  # Radius for positioning the days around the center

# Position each day in a circular layout around the "Week" node
for i, day in enumerate(days):
    angle = i * angle_step
    pos[day] = (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle))

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", edge_color="gray", arrows=False, font_size=10)

# Display the plot
plt.title("Week Layout with Days")
plt.show()
