from Datasets.swda.swda import Transcript
import pandas as pd
import glob
import networkx

"""
Author: Arvid L. 2020, UvA
class GraphCreator takes as input a set 
of command line arguments inside an ArgumentParser object
and converts a dataset into three graph-representations 
used for training, validating and testing a graph neural network.
The class GraphVisualizer can be used to visualize the 
resulting graphs.

"""

class GraphCreator(object):

    def __init__(self, cli_args):
        self.dataset = cli_args.dataset
        self.train_split = cli_args.split[0]
        self.validation_split = cli_args.split[1]
        self.test_split = cli_args.split[2]
        self.graph_type = cli_args.type
        self.graph_name = cli_args.graph_name
        self.random_seed = cli_args.random_seed

        self.train_graph = None
        self.validation_graph = None
        self.test_graph = None

        self.out_folder = self.graph_name + "_folder"
        self.file_name = self.graph_name + ".pkl"

        print("\nGraphCreator instantiated as:")
        print(',\n'.join("%s: %s" % item for item in vars(self).items()))
        print("*" * 30)

    # Extracts three lists of file names for train/valid and test
    # respectively from the chosen dataset using random_seed to
    # shuffle data
    def selectDialogues(self, dataset):
        pass

    # Makes a graph of SwDA dialogues based on the
    # arguments in self.graph_type
    def constructSwdaGraphs(self, list_of_dialogues):
        pass

    # Makes a graph of MRDA dialogues based on the
    # arguments in self.graph_type
    def constructMRDAGraphs(self, list_of_dialogues):
        pass

    # Makes a graph of Mantis dialogues based on the
    # arguments in self.graph_type
    def constructMantisGraphs(self, list_of_dialogues):
        pass

    # Makes a graph of MSDialog dialogues based on the
    # arguments in self.graph_type
    def constructMSDialogGraphs(self, list_of_dialogues):
        pass

    def graph2GraphSAGEFormat(self, graph):

    def createGraph(self):

        print(f"Created graph: {self.graph_name}")


    def saveGraphOnDisk(self):

        print(f"Saved graph in: {self.out_folder}\{self.file_name}")
