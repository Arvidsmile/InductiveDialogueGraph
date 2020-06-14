"""
Tutorial on how to load data and work with the datastructure.
[] Try it out using SwDA.csv first,
[] Then test it with added TF-IDF features for each utterance.

"""

from stellargraph import StellarGraph
import pandas as pd
from pyvis.network import Network as visNet
import numpy as np

#1. Define a graph using a dataframe representing the edges
# source "is connected to" target
square_edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"],
     "target": ["b", "c", "d", "a", "c"]}
)

square = StellarGraph(edges = square_edges)

#2. Add features to each node
feature_array = np.array([[-1, 0.4], [2, 0.1], [-3, 0.9], [4, 0]])


nt = visNet()
netxG = square.to_networkx(feature_attr = "utterance_embeddings")
print(netxG.graph)
nt.from_nx(netxG)

print(nt.nodes)

for node in nt.nodes:
    node["title"] = "hello"

print(nt)
nt.show("testing.html")

