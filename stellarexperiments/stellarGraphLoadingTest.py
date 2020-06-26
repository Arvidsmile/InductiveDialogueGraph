"""
Tutorial on how to load data and work with the datastructure.
[] Try it out using SwDA.csv first

"""

from stellargraph import StellarGraph, IndexedArray
import networkx as nx
import pandas as pd
from pyvis.network import Network as visNet
import numpy as np

#1. Define a graph using a dataframe representing the edges
# source "is connected to" target
square_edges = pd.DataFrame(
    {"source": ["Actor1", "Actor2", "u1", "u2", "Actor1"],
     "target": ["Actor2", "u1", "u2", "Actor1", "u1"]}
)

#2. Add features to each node
feature_array = np.array([[-1, 0.4], [2, 0.1], [-3, 0.9], [4, 0]])
square_nodes = IndexedArray(feature_array, index = ["Actor1", "Actor2", "u1", "u2"])

# 3. Build the graph
square = StellarGraph(edges = square_edges, nodes = square_nodes)

# 4. Visualize graph
nt = visNet()
netxG = square.to_networkx(feature_attr = "Utterance_embeddings")
# print(netxG.graph)
nt.from_nx(netxG)

print(nt.nodes)
# print(nt.edges)

for node in nt.nodes:
    node["title"] = "hello"

print(nt)
nt.show("testing.html")

