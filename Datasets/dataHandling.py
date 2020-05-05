import pandas as pd
from Datasets.swda.swda import Transcript
import os

"""
Author: Arvid L. 2020, UvA
class DataHandler is used to work directly with the 
datasets as provided by their respective authors. It 
provides the GraphCreator with a unified datastructure 
that works across all datasets and makes it easy to build 
graphs using panda dataframes as input.

"""

class DataHandler(object):

    def __init__(self):
        pass

    # Function checks folder structure and looks for the file
    # with the right type
    def doesCSVExist(self, dataset):
        pass

    # Goes through existing .csv file of dataset and
    # returns its length, prompts user to create dataset if
    # it does not exist already in .csv format
    @staticmethod
    def lengthDataset(dataset):
        pass

    # Extracts three lists of file names for train/valid and test
    # respectively from the chosen dataset using random_idx to
    # select data, if the .csv file for the dataset does not exist,
    # it prompts the user to run the class/script as main
    @staticmethod
    def selectDialogues(dataset, random_idx, trainsplit, valsplit, testsplit, type = 'train'):
        # 0. Check that the pre-processed .csv exists, else cancel the script
        # and prompt user to run
        # InductiveDialogueGraph$ python -m Datasets.dataHandling
        pass

    # The following functions are used (ideally) only once to
    # convert the data from the original datasets into a more manageable
    # format where we can read specific rows as desired from a .csv file.
    def swda2csv(self):
        pass

    def mrda2csv(self):
        pass

    def mantis2csv(self):
        pass

    def msdialog2csv(self):
        pass

    csv_func = {'swda' : swda2csv,
                'mrda' : mrda2csv,
                'mantis' : mantis2csv,
                'msdialog' : msdialog2csv}

    # Takes as input, the location of the folder for
    # corresponding graph and creates a .csv representation
    # that may be used in the future, or loads from disk if
    # it already exists, identified by the random_seed value
    # and the split.
    def data2csv(self):
        pass

# Run as main script to create the .csv files, should
# they not already exist
if __name__ == '__main__':
    dHandler = DataHandler()
    dataset = input("Which dataset do you want to create a .csv representation of?\n[swda, mrda, msdialog, mantis]\n> ")
    print(dataset)