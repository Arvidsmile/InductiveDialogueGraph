"""
Author: Arvid Lindström
    arvid.lindstrom@student.uva.nl (12365718)
    arvid.lindstrom@kpn.com
    arvid.lindstrom@gmail.com

Main file for creating a graphical representation
of one of the four datasets:
    1. MANTIS_Intent:
        Manually annotated user intent labels of
        information-seeking conversations from the community
        question-answering portal Stack Exchange.
    2. MRDA:
        The ICSI MRDA Corpus consists of hand-annotated dialog act,
        adjacency pair, and hotspot labels for the 75 meetings in
        the ICSI meeting corpus.
    3. MSDialog:
        Dialog dataset of question answering (QA)
        interactions between information seekers and
        answer providers from an online forum on Microsoft products.
    4. SwDA:
        Extends the Switchboard-1 Telephone Speech Corpus, Release 2
        with turn/utterance-level dialog-act tags

The script is run as follows:
python -m graphCreation.main --dataset [mantis, mrda, msdialog, swda]
    --type [path-graph, unique-utterance, actor-nodes]
    --split "0.8 0.1 0.1"
    --graph_name "default_graph_name"
    --random_seed 1337

--dataset defines which of the four datasets we want to work on
--type defines the kind of graph we want to create, which internally depends
    on the dataset (and its available variable fields such as actor or timestamp)
--split is a string structured as "train/valid/test" and corresponds to the
    proportion of dialogues to set aside for training, validating and testing
    a model, respectively.
--output_path defines where to store the output of the graph in pickle format
"""

import sys
import argparse
from graphCreation.graphCreator import GraphCreator
from graphCreation.graphVisualizer import GraphVisualizer

# This utility function checks the command line arguments
# that instantiates a GraphCreator object
def confirmCommandLineArguments(args):

    # Confirm that --split argument contains three floats that sum to one
    assert len(args.split) == 3, "--split contains more/fewer than three values (should be floats: train valid test)"
    assert sum(args.split) == 1.0, "--split train/val/test proportions do not add up to 1.0"

    if args.graph_name is None:
        print(f"\nNotice: Output folder/file name not specified, set to default \"{args.graph_name}\"")

    print("\nArguments successfully parsed as:")
    for arg in vars(args):
        print('--' + arg, ':', getattr(args, arg))
    print("*" * 30)


if __name__ == '__main__':
    #0. Parse command line arguments
    usage = "python -m graphCreation.main --dataset [mantis, mrda, msdialog, swda]" + \
        "\n --type [path-graph, unique-utterance, actor-nodes]" + \
        "\n --split 0.8(train) 0.1(valid) 0.1(test)" + \
        "\n --graph_name default_graph_name"
    parser = argparse.ArgumentParser(usage = usage)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=['swda', 'mrda', 'msdialog', 'mantis'],
        help="Select which dataset to create graph-representation of.",
        default='swda'
    )

    parser.add_argument(
        "--split",
        type = float,
        help = "Enter three floats corresponding to train, valid and testing proportions",
        default = [0.8, 0.1, 0.1],
        nargs = "+"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=['path-graph', 'unique-utterance', 'actor-nodes'],
        help="Select the graph creation schema(s) to use",
        default= ['path-graph'],
        nargs = "*"
    )

    parser.add_argument(
        "--graph_name",
        type=str,
        help="Used for destination folder to save graph file as both pickle (.pkl) and .json",
        default='default_graph_name'
    )

    parser.add_argument(
        "--random_seed",
        type = int,
        help = "Enter a seed for reproducibility",
        default = 1337
    )

    # 0. Check that input has been done correctly
    args = parser.parse_args()
    try:
        confirmCommandLineArguments(args)
    except AssertionError as parse_error:
        print(f"Incorrect input arguments: {parse_error}")
        sys.exit()

    #1. Create graphCreator
    gCreator = GraphCreator(args)

    #2. Create graph representation
    gCreator.createGraphNetworkx()

    #3. Convert graph to format expected by GraphSAGE
    gCreator.createGraphJson()

    #3. Store graph on disk in both networkx and .json format (GraphSAGE format)
    gCreator.saveGraph()

    #4. Show the graph
    GraphVisualizer(gCreator.train_graph)
