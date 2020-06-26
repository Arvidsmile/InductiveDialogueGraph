# Graph related imports
from stellargraph import StellarGraph, IndexedArray, StellarDiGraph
from stellargraph.layer import HinSAGE, DirectedGraphSAGE
from stellargraph.mapper import DirectedGraphSAGENodeGenerator
import stellargraph as sg

# ML imports
import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop, Adam

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

# General utilities
import sys
import argparse
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil

def load_graph_data(dataframe, embeddings, name = "default", testing = False, num_test = 100):

    actor_indeces = []
    actor_features = []
    utterance_indeces = []
    utterance_features = []
    source_edges = []
    target_edges = []

    if testing:
        num_dialogues = num_test
    else:
        num_dialogues = len(dataframe['Dialogue ID'].unique())

    print("Building graph, 1 dialogue at a time...")
    for dialogueID in tqdm(dataframe['Dialogue ID'].unique()[0:num_dialogues]):
        dialogue = dataframe[dataframe["Dialogue ID"] == dialogueID]

        # Loop through all utterances of the dialogue
        for rowidx in range(len(dialogue)):
            row = dialogue.iloc[rowidx]

            # 0. Add actor index-feature if it does not already exist
            actor_idx = f"{row.Actor}_{dialogueID}"
            if actor_idx not in actor_indeces:
                actor_indeces.append(actor_idx)
                if len(actor_features) == 0:
                    # Create new numpy array of actor features
                    actor_features = np.random.normal(0.0, 1.0, [1, 1024])
                else:
                    # Concatenate features to already existing array
                    actor_features = np.concatenate((actor_features,
                                                     np.random.normal(0.0, 1.0, [1, 1024])),
                                                    axis = 0)
            # 1. Add utterance index-feature (ELMo embeddings)
            utt_idx = f"u_dID{dialogueID}_#{rowidx}"
            utterance_indeces.append(utt_idx)
            # To iterate over the ELMo embeddings we use the index list of the
            # dataset, indexed by the row of the dialogue we are currently parsing
            if len(utterance_features) == 0:
                utterance_features = np.array([embeddings[dialogue.index[rowidx]]])
            else:
                utterance_features = np.concatenate((utterance_features,
                                                     np.array([embeddings[dialogue.index[rowidx]]])),
                                                    axis = 0)

            # 2. Build edges. If this is the first row of a dialogue,
            # begin by drawing an edge from the "START-Node" (source)
            # to the current utterance index (target)
            if rowidx == 0:
                source_edges.append("START-Node")
                target_edges.append(utt_idx)

            # 3. Construct remaining edges.
            # 3.1 Actor to the utterance
            source_edges.append(actor_idx)
            target_edges.append(utt_idx)
            # 3.2 Utterance to the next utterance
            if (rowidx + 1) != len(dialogue):
                source_edges.append(utt_idx)
                target_edges.append(f"u_dID{dialogueID}_#{rowidx + 1}")
            # 3.3 Utterance to all actors
            for actor in dialogue['Actor'].unique():
                all_actor_idx = f"{actor}_{dialogueID}"
                source_edges.append(utt_idx)
                target_edges.append(all_actor_idx)

    # HinSAGE (Does not support directionality) :( ..bad
    # Create IndexedArrays of actors and utterances
    # actor_nodes = IndexedArray(actor_features, actor_indeces)
    # utterance_nodes = IndexedArray(utterance_features, utterance_indeces)
    # ... Todo: ---> Add start_node index array here if u want hinsage.. <---

    # GraphSAGE (Does not support modelling nodes of different kind) ..less bad
    start_features = np.random.normal(0.0, 1.0, [1, 1024])
    start_index = "START-Node"
    node_features = np.concatenate((actor_features,
                                    utterance_features,
                                    start_features), axis = 0)
    node_indeces = actor_indeces + utterance_indeces + [start_index]
    nodes = IndexedArray(node_features, node_indeces)

    edges = pd.DataFrame({
        "source" : source_edges,
        "target" : target_edges
    })
    # HinSAGE:
    # full_graph = StellarDiGraph({"start" : start_node,
    #                              "actor" : actor_nodes,
    #                              "utterance" : utterance_nodes},
    #                             edges)

    # GraphSAGE:
    full_graph = StellarDiGraph(nodes, edges)

    targets = pd.Series(dataframe['Dialogue Act'].tolist()[0:len(utterance_indeces)],
                        index = utterance_indeces)

    print("Check if graph has all properties required for ML/Inference...")
    full_graph.check_graph_for_ml(expensive_check = True)
    print("Check successful.")
    print(full_graph.info())
    print("---- Graph Creation Finished ----")

    netx_graph = full_graph.to_networkx(feature_attr = 'utterance_embedding')
    # Save graphs for later use.
    if testing:
        pickle.dump((netx_graph, targets), open(f"visualizeGraph/test_{name}_netx.pickle", "wb"))
        pickle.dump((full_graph, targets), open(f"createdGraphs/test_{name}_graph.pickle", "wb"))
    else:
        pickle.dump((netx_graph, targets), open(f"visualizeGraph/{name}_netx.pickle", "wb"))
        pickle.dump((full_graph, targets), open(f"createdGraphs/{name}_graph.pickle", "wb"))

    return full_graph, targets

def isInSplit(node, dialogueIDs):

    # Is it an actor? (e.g. user_33 or B_412)
    if len(node.split('_')) == 2:
        return int(''.join(list(filter(str.isdigit, node)))) in dialogueIDs
    # Is it an utterance? (e.g. u_dID33_#5)
    if len(node.split('_')) == 3:
        return int(''.join(list(filter(str.isdigit, node.split('_')[1])))) in dialogueIDs
    else:
        return False

# Function takes sampled dialogueIDs and
def splitByDialogue(dialogueIDs, full_graph):

    node_indeces = []

    # 1. Extract the start node
    # node_indeces.append(full_graph.nodes_of_type("start")[0])
    node_indeces.append("START-Node")

    # 2. Extract actor and utterance nodes from chosen dialogues
    node_indeces += [node for node in full_graph.nodes() if isInSplit(node, dialogueIDs)]

    node_indeces = pd.Index(node_indeces)

    # Make a separate list of utterance indeces to be used to index the target labels
    utterance_indeces = [node_idx for node_idx in node_indeces if node_idx[0:5] == 'u_dID']
    utterance_indeces = pd.Index(utterance_indeces)

    return node_indeces, utterance_indeces

def createBinarizer(dataset, multi = False):

    # This function goes through the targets equal to the
    # chosen utterance_indeces for either training or inference
    # and creates labels for training and also returns the binarizer
    # so that we may retrieve the labels afterwards.
    labelset = []
    if multi:
        for DAs in dataset['Dialogue Act']:
            labels = set(DAs.split('-'))
            labelset.append(labels)
        mlb = MultiLabelBinarizer()
        mlb.fit(labelset)
        return mlb
    else:
        for DAs in dataset['Dialogue Act']:
            labelset.append(DAs)
        binarizer = LabelBinarizer()
        binarizer.fit(labelset)
        return binarizer

def binarizeMultiLabelDataset(binarizer, chosenUtterances):
    labelset = []

    for DAs in chosenUtterances:
        labels = set(DAs.split('-'))
        labelset.append(labels)

    return binarizer.transform(labelset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help = "Which dataset?",
                        type = str,
                        default = "SwDA")
    parser.add_argument("--expname",
                        help = "Name this experiment",
                        type = str,
                        default = "MyExperiment")
    parser.add_argument("--testing",
                        help = "Are we in testing mode?",
                        type = int,
                        default = 0)
    parser.add_argument("--batch_size",
                        help = "Size of each batch of utterances.",
                        type = int,
                        default = 32)
    parser.add_argument("--epochs",
                        help = "Number of epochs for training",
                        type = int,
                        default = 50)
    parser.add_argument("--kfold",
                        help="Number of folds to do training/testing in",
                        type=int,
                        default=10)

    parser.add_argument("--num_test",
                        help="Number of dialogues for testing",
                        type=int,
                        default=100)

    args = parser.parse_args()

    # --------------------------------------------
    # --- 1. Create graphs and indexed labels ----
    # --------------------------------------------
    if args.dataset in ['SwDA', 'MRDA']:

        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")

        full_graph, targets = load_graph_data(dataset,
                                              embeddings,
                                              args.dataset,
                                              args.testing,
                                              args.num_test)

    elif args.dataset in ['MANtIS', 'MSDialog']:
        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")

        full_graph, targets = load_graph_data(dataset,
                                              embeddings,
                                              args.dataset,
                                              args.testing,
                                              args.num_test)

    else:
        print("Please enter a valid dataset: SwDA | MRDA | MANtIS | MSDialog | testing")
        sys.exit()

    # --------------------------------------------
    # --- 2. Create graphs and indexed labels ----
    # --------------------------------------------
    kFold = KFold(n_splits = args.kfold)
    current_fold = 1

    # Split the graph based on DialogueID, such that nodes sampled are
    # always from complete dialogues.
    if args.testing:
        dialogues = dataset["Dialogue ID"].unique()[0:args.num_test]
    else:
        dialogues = dataset["Dialogue ID"].unique()

    for training_split, inference_split in kFold.split(dialogues):

        train_node_indeces, train_utterance_indeces = splitByDialogue(training_split,
                                                                      full_graph)
        inference_node_indeces, inference_utterance_indeces = splitByDialogue(inference_split,
                                                                        full_graph)

        print(f"Split from fold {current_fold}:")
        print("Train on: ", train_utterance_indeces.shape, " nodes.")
        # Sample subgraph used to train/validate from the full graph
        graph_train_sampled = full_graph.subgraph(train_node_indeces)
        # print(graph_train_sampled.info())

        # -----------------------------------------------------
        # -- 3. Binarize training labels depending on dataset -
        # -----------------------------------------------------
        if args.dataset in ['SwDA', 'MRDA']:

            target_encoding = createBinarizer(dataset, multi = False)
            train_one_hot = target_encoding.transform(targets[train_utterance_indeces])
            print(train_one_hot.shape)

        else: #<-- we must be in either MANtIS or MSDialog
            target_encoding = createBinarizer(dataset, multi=True)
            train_multi_hot = binarizeMultiLabelDataset(target_encoding, targets[train_utterance_indeces])
            # train_multi_hot = target_encoding.transform(targets[train_utterance_indeces])
            print(train_multi_hot.shape)

        # -----------------------------------------------------
        # -- 4. Build GraphSAGE model and generator for train -
        # -----------------------------------------------------
        batch_size = args.batch_size
        in_samples = [5, 1, 1]
        out_samples = [5, 1, 1]
        generator = DirectedGraphSAGENodeGenerator(graph_train_sampled,
                                                   batch_size,
                                                   in_samples,
                                                   out_samples)

        if args.dataset in ['SwDA', 'MRDA']:
            assert (len(train_utterance_indeces) == len(train_one_hot))
            train_gen = generator.flow(train_utterance_indeces, train_one_hot, shuffle = True)
        else:
            assert (len(train_utterance_indeces) == len(train_multi_hot))
            train_gen = generator.flow(train_utterance_indeces, train_multi_hot, shuffle = True)

        # -----------------------------------------------------
        # -- 5. Specify machine learning model ----------------
        # -----------------------------------------------------
        # Note: For multi_hot_labels we require sigmoid activation
        # and binary crossentropy loss.

        graphsage_model = DirectedGraphSAGE(
            layer_sizes=[128, 128, 128], generator=generator, bias=True, dropout=0.5,
        )
        # Define input and output tensors of graphsage model
        x_inp, x_out = graphsage_model.in_out_tensors()
        # Define the prediction layer
        if args.dataset in ['SwDA', 'MRDA']:
            first_layer = layers.Dense(units=1024, activation="relu")(x_out)
            prediction = layers.Dense(units=train_one_hot.shape[1], activation="softmax")(first_layer)
        else:
            first_layer = layers.Dense(units=1024, activation="relu")(x_out)
            prediction = layers.Dense(units=train_multi_hot.shape[1], activation="sigmoid")(first_layer)
        # -----------------------------------------------
        # -- 5. Train machine learning model ------------
        # -----------------------------------------------
        model = Model(inputs=x_inp, outputs=prediction)

        if args.dataset in ['SwDA', 'MRDA']:
            model.compile(
                optimizer=optimizers.Adam(lr=0.005),
                loss=losses.categorical_crossentropy,
                metrics=["acc"],
            )
        else:
            model = Model(inputs=x_inp, outputs=prediction)
            model.compile(
                optimizer=optimizers.Adam(lr=0.005),
                loss=losses.binary_crossentropy,
                metrics=["acc"],
            )

        history = model.fit(
            train_gen, epochs = args.epochs,
            verbose = 1, shuffle = False
        )

        # Plot training graph
        sg.utils.plot_history(history, return_figure = True)

        dir = f'training_plots/{args.expname}_{args.dataset}'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        if args.testing:
            plt.savefig(f"{dir}/{args.dataset}_test_training_fold:{current_fold}.png")
        else:
            plt.savefig(f"{dir}/{args.dataset}_training_fold:{current_fold}.png")

        # -----------------------------------------------
        # -- 6. Perform testing on unseen dialogues -----
        # -----------------------------------------------

        generator = DirectedGraphSAGENodeGenerator(full_graph,
                                                   batch_size,
                                                   in_samples,
                                                   out_samples)

        # print("Inference on: ", inference_utterance_indeces.shape, " nodes.")
        # graph_inference_sampled = full_graph.subgraph(inference_node_indeces)
        # print(graph_inference_sampled.info())
        if args.dataset in ['SwDA', 'MRDA']:
            inference_one_hot = target_encoding.transform(targets[inference_utterance_indeces])
            assert (len(inference_utterance_indeces) == len(inference_one_hot))
            inference_gen = generator.flow(inference_utterance_indeces, inference_one_hot, shuffle = True)
            # print(inference_one_hot.shape)
        else:
            inference_multi_hot = binarizeMultiLabelDataset(target_encoding, targets[inference_utterance_indeces])
            assert (len(inference_utterance_indeces) == len(inference_multi_hot))
            inference_gen = generator.flow(inference_utterance_indeces, inference_multi_hot, shuffle=True)
            # print(inference_multi_hot.shape)

        # print("*" * 30)

        hold_out_predictions = model.predict(inference_gen)

        hold_out_loss_accuracy = model.evaluate(inference_gen)
        print(f"\nFold[{current_fold}] Hold Out Set Metrics:")
        for name, val in zip(model.metrics_names, hold_out_loss_accuracy ):
            print("\t{}: {:0.4f}".format(name, val))

        # Print model on the very last fold
        if current_fold == args.kfold:
            print(model.summary())

        current_fold += 1

        # -----------------------------------------------
        # -- 7. Todo: Record testing metrics, f1micro, f1macro, precision, accuracy -----
        # -----------------------------------------------

