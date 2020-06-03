import pandas as pd
from Datasets.swda.swda import Transcript
import os
import sys
import glob
from tqdm import tqdm

"""
Author: Arvid L. 2020, UvA
class DataHandler is used to work directly with the 
datasets as provided by their respective authors. It 
provides the GraphCreator with a unified datastructure 
that works across all datasets and makes it easy to build 
graphs using panda dataframes as input.

"""

# Required globals
swda_tag_to_name = \
    {
        'ft': 'Thanking',
        'bk': 'Response Acknowledgement',
        'qy^d': 'Declarative Yes-No Question',
        'bf': 'Summarize/reformulate',
        '^q': 'Quotation',

        'qw': 'Wh-Question',
        'ar': 'Reject',
        'fc': 'Conventional-Closing',
        'qy': 'Yes-No-Question',
        'b^m': 'Repeat-Phrase',

        'ba': 'Appreciation',
        '^g': 'Tag-Question',
        'sd': 'Statement-Non-Opinion',
        'br': 'Signal-Non-Understanding',
        'qo': 'Open-Question',

        '^h': 'Hold before answer/agreement',
        'na': 'Affirmative non-yes answer',
        'oo_co_cc': 'Offers, Options, Commits',
        'qw^d': 'Declarative Wh-Question',
        'no': 'Other answers',

        'x': 'Non-verbal',
        'fp': 'Conventional-opening',
        'b': 'Acknowledge (Backchannel)',
        'arp_nd': 'Dispreferred answers',
        'bh': 'Backchannel in question form',

        'h': 'Hedge',
        'nn': 'No Answers',
        '%': 'Uninterpretable',
        't1': 'Self-talk',
        'fo_o_fw_"_by_bc': 'Other',

        'aa': 'Agree/Accept',
        'aap_am': 'Maybe/Accept-part',
        'bd': 'Downplayer',
        'fa': 'Apology',
        'ny': 'Yes answers',

        'ad': 'Action-directive',
        'qh': 'Rhetorical-Question',
        'qrr': 'Or-Clause',
        '^2': 'Collaborative Completion',
        'ng': 'Negative non-no answers',

        'sv': 'Statement-Opinion',
        't3': '3rd-party-talk',
        '+': 'Segment',
    }


class DataHandler(object):

    @staticmethod
    def promptCSVCreation(dataset):
        print(f"Error: {dataset}-Dataset has not yet been converted to .csv format.\n",
            "Please run file Datasets/dataHandling.py as main: \n",
              "1. Navigate to InductiveDialogueGraph\n",
              "2. Run the command: \n",
              "python -m Datasets.dataHandling")

    # Function checks folder structure and looks for the file
    # with the right type
    @staticmethod
    def doesCSVExist(dataset):
        return os.path.exists(f'../CSVData/{dataset}.csv')

    # Goes through existing .csv file of dataset and
    # returns its number of unique conversations,
    # prompts user to create dataset if
    # it does not exist already in .csv format
    @staticmethod
    def lengthDataset(dataset):
        if not DataHandler.doesCSVExist(dataset):
            DataHandler.promptCSVCreation(dataset)
            sys.exit()
        else:
            file = open(f"../CSVData/{dataset}length.txt", 'r')
            return int(file.read())

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # The following functions are used (ideally) only once to
    # convert the data from the original datasets into a more manageable
    # format where we can read specific rows as desired from a .csv file.
    # At the end of function, saved the length of dataset as a variable in a .txt file
    # for later usage
    def swda2csv(self):
        dataframe_list = []
        # Find all names of transcripts in swda corpus
        file_names = glob.glob("Datasets/swda/swda/sw**utt/*")

        for i in tqdm(range(len(file_names))):
            # Store name ID of conversation
            id = file_names[i]

            # id = id[7:14] + '-' + id[15:27]

            # Read file and
            # go through all utterances
            dialogue = Transcript(id, 'Datasets/swda/swda/swda-metadata.csv')

            for utter in dialogue.utterances:
                utter_DA = swda_tag_to_name[utter.damsl_act_tag()]  # <-- add dictionary cleanup
                utterance_text = ' '.join(utter.pos_words())
                actor = utter.caller
                dataframe_list.append({'Dialogue ID': i, #<-- conversations uniquely identified by number {0, 1, .. M}
                                       'Actor': actor,
                                       'Utterance': utterance_text,
                                       'Dialogue Act': utter_DA})

        df = pd.DataFrame(dataframe_list)

        if os.path.exists('../CSVData/swda.csv'):
            os.remove('../CSVData/swda.csv')

        if os.path.exists('../CSVData/swdalength.txt'):
            os.remove('../CSVData/swdalength.txt')

        df.to_csv('../CSVData/swda.csv')
        # Note down number of unique dialogues
        f = open("../CSVData/swdalength.txt", 'w+')
        f.write(str(len(file_names)))
        f.close()

    def mrda2csv(self):
        raise NotImplementedError

    def mantis2csv(self):
        raise NotImplementedError

    def msdialog2csv(self):
        raise NotImplementedError

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self, dataset):

        if not DataHandler.doesCSVExist(dataset):
            DataHandler.promptCSVCreation(dataset)
            sys.exit()

        # Read the desired .csv file
        self.df = pd.read_csv(f"../CSVData/{dataset}.csv")

        self.csv_func = {'swda': self.swda2csv,
                         'mrda': self.mrda2csv,
                         'mantis': self.mantis2csv,
                         'msdialog': self.msdialog2csv}

    # Extracts chosen dataset using indeces to
    # select data
    def selectDialogues(self, indeces):
        return self.df.loc[self.df["Dialogue ID"].isin(indeces)]

# Run as main script to create the .csv files, should
# they not already exist
if __name__ == '__main__':
    dHandler = DataHandler()
    dataset = input("Which dataset do you want to create a .csv representation of?\n[swda, mrda, msdialog, mantis]\n> ")
    print(f"Building CSVData/{dataset}.csv")
    dHandler.csv_func[dataset]()
