
import pandas as pd
import glob
import networkx
from Datasets.dataHandling import DataHandler
from tqdm import tqdm
import os

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
        self.random_seed = cli_args.random_seed
        self.graph_name = cli_args.graph_name + '_seed(' \
            + str(self.random_seed) + ')_split(' + str(self.train_split) + \
            '|' + str(self.validation_split) + '|' + str(self.test_split) + ')'

        self.train_graph = None
        self.validation_graph = None
        self.test_graph = None

        print("\nGraphCreator instantiated as:")
        print(',\n'.join("%s: %s" % item for item in vars(self).items()))
        print("*" * 30)

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

    # Some static members
    construction_func = {'swda' : constructSwdaGraphs,
                         'mrda' : constructMRDAGraphs,
                         'mantis' : constructMantisGraphs,
                         'msdialog' : constructMSDialogGraphs}

    # Takes graph in networkx format and turns it into the format
    # expected by the GraphSAGE library
    def graph2GraphSAGEFormat(self, graph):
        pass

    # Function goes through folder structure
    # and looks for a folder that is named after
    # the self.graph_name variable
    def doesGraphExist(self):
        pass

    # Creates three graphs used for training, testing and validation based on splits
    # provided in train_split, val_split and test_split (.self)
    # Upon first creation, will prompt user if graph should be loaded from
    # disk, should it exist. If a new graph is created, user will be asked
    # if it should be saved to disk for faster usage in future. Graphs will
    # then be accessible from members self.train/val/test_graph
    def createGraph(self):

        # 0. Check if graph has already been saved from a previous run
        # (networkx graph exists in pickled format)
        if self.doesGraphExist():
            # 0.1 Load graph from networkx format
        else:
            # 1. If it does not exists, we need to create it based on the
            # random seed.
            random_indeces = list(range(DataHandler.lengthDataset(self.dataset)))

        # 2. Convert graph into format expected by graphSAGE and store it
        # in the same folder in the .json format


        print(f"Created graph: {self.graph_name}")



    def saveGraphOnDisk(self):

        # Specify where to save graph in networkx format
        self.out_folder = self.graph_name + "_folder"
        self.file_name = self.graph_name + ".pkl"

        print(f"Saved graph in: {self.out_folder}\{self.file_name}")
