import argparse
import networkx as nx
import pickle
from pyvis.network import Network as visNet
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help = "Which dataset?",
                        type = str,
                        default = "SwDA")

    args = parser.parse_args()
    netxgraph = pickle.load( open(f"{args.dataset}.pickle", "rb"))

    # Make graph visualization
    nt = visNet()
    print("Building pyvis-graph from networkx...")
    nt.from_nx(netxgraph)
    print("Done building visualization graph.")

    # for node in nt.nodes:
    #     node["title"] = "hello"
    #     node["utterance_embedding"] = str(node["utterance_embedding"])
    #     print(node)

    nt.show(f"{args.dataset}.html")
