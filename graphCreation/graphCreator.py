
import pandas as pd
import glob
import networkx
from Datasets.dataHandling import DataHandler
from tqdm import tqdm
import os
import shutil
import random

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

    # Makes a graph of SwDA dialogues based on the
    # arguments in self.graph_type
    def constructSwdaGraphs(self, df_dialogues):

        print("Received dialogues:")
        print(df_dialogues.head(), len(df_dialogues))

        raise NotImplementedError

    # Makes a graph of MRDA dialogues based on the
    # arguments in self.graph_type
    def constructMRDAGraphs(self, df_dialogues):
        raise NotImplementedError

    # Makes a graph of Mantis dialogues based on the
    # arguments in self.graph_type
    def constructMantisGraphs(self, df_dialogues):
        raise NotImplementedError

    # Makes a graph of MSDialog dialogues based on the
    # arguments in self.graph_type
    def constructMSDialogGraphs(self, df_dialogues):
        raise NotImplementedError

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
    def createGraphNetworkx(self):
        construction_func = {'swda': self.constructSwdaGraphs,
                             'mrda': self.constructMRDAGraphs,
                             'mantis': self.constructMantisGraphs,
                             'msdialog': self.constructMSDialogGraphs}

        # 0. Check if graph has already been saved from a previous run
        # (networkx graph exists in pickled format)
        if self.doesGraphExist():
            # 0.1 Load graph from networkx format
            raise NotImplementedError
        else:
            # 1. If it does not exists, we need to create it based on the
            # random seed.
            random_indeces = list(range(DataHandler.lengthDataset(self.dataset)))
            random.shuffle(random_indeces)
            # 2. Create graphs for each of the splits
            dHandler = DataHandler(self.dataset)
            # --train
            train_indeces = random_indeces[:int(len(random_indeces) * self.train_split)]
            self.train_graph = construction_func[self.dataset](dHandler.selectDialogues(train_indeces))
            # --validation
            val_indeces = random_indeces[int(len(random_indeces) * self.train_split):\
                          int(len(random_indeces) * self.train_split) + int(len(random_indeces) * self.validation_split)]
            self.validation_graph = construction_func[self.dataset](dHandler.selectDialogues(val_indeces))
            # --testing
            test_indeces = random_indeces[int(len(random_indeces) * self.train_split) + \
                                          int(len(random_indeces) * self.validation_split):]
            self.test_graph = construction_func[self.dataset](dHandler.selectDialogues(test_indeces))

    # Takes graph in networkx format and turns it into the format
    # expected by the GraphSAGE library
    def createGraphJson(self):
        raise NotImplementedError

    def saveGraph(self):

        #0. Specify where to save graph
        if not os.path.exists('../SavedGraphs'):
            os.mkdir('../SavedGraphs')

        topfolder = f'../SavedGraphs/{self.graph_name}'

        #1. Should the folder already exist, clear all its content and recreate it
        if os.path.exists(topfolder):
            shutil.rmtree(topfolder)

        os.mkdir(f'../SavedGraphs/{topfolder}')

        # Networkx name
        self.file_name = self.graph_name + ".pkl"

        # GraphSAGE format name(s)


        print(f"Saved graph in: {topfolder}")

        raise NotImplementedError

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

        print("GraphCreator instantiated as:")
        print(',\n'.join("%s: %s" % item for item in vars(self).items()))
        print("*" * 30)

        self.train_graph = None
        self.validation_graph = None
        self.test_graph = None

        # Set the seed used to generate data-selection splits
        random.seed(self.random_seed)