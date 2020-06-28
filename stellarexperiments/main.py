# Graph related imports
from stellargraph import StellarGraph, IndexedArray, StellarDiGraph
from stellargraph.layer import HinSAGE, DirectedGraphSAGE
from stellargraph.layer import MeanAggregator, MeanPoolingAggregator, MaxPoolingAggregator, AttentionalAggregator
from stellargraph.mapper import DirectedGraphSAGENodeGenerator
import stellargraph as sg

# ML imports
import keras

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
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import os
import shutil
os.environ['KMP_WARNINGS'] = '0'

def load_graph_data(dataframe, embeddings, name = "default",
                    testing = False, num_test = 100, using_start = False):

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
            if using_start and rowidx == 0:
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

    # GraphSAGE (Does not support modelling nodes of different kind) ..less bad

    if using_start:
        start_features = np.random.normal(0.0, 1.0, [1, 1024])
        start_index = "START-Node"
        node_features = np.concatenate((actor_features,
                                        utterance_features,
                                        start_features), axis = 0)
        node_indeces = actor_indeces + utterance_indeces + [start_index]
    else:
        node_features = np.concatenate((actor_features,
                                        utterance_features), axis = 0)
        node_indeces = actor_indeces + utterance_indeces

    nodes = IndexedArray(node_features, node_indeces)

    edges = pd.DataFrame({
        "source" : source_edges,
        "target" : target_edges
    })

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
def splitByDialogue(dialogueIDs, full_graph, using_start = False):

    node_indeces = []

    # 1. Extract the start node
    if using_start:
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

def chooseAggregator(aggregator):
    return {"mean": MeanAggregator,
            "meanpool": MeanPoolingAggregator,
            "maxpool": MaxPoolingAggregator,
            "attention": AttentionalAggregator}[aggregator]

def plotTSNE(embeddings, DAlabels, dir, plotname, show = False):
    print("Plot the tsne-representation of the embeddings")
    trans = TSNE(n_components=2)

    # Create TSNE embeddings
    emb_transformed = pd.DataFrame(trans.fit_transform(embeddings))
    # print(DAlabels)
    integerLabels = pd.factorize(DAlabels)[0]
    # print(integerLabels)
    emb_transformed["label"] = integerLabels
    emb_transformed["Dialogue Act"] = DAlabels.tolist()
    # print(emb_transformed)

    # Plot a scatterplot for each dialogue act category
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(10, 10))
    label_colors = iter(cm.rainbow(np.linspace(0, 1, len(emb_transformed["label"].unique()))))
    for cat, color in tqdm(zip(emb_transformed["label"].unique(), label_colors)):
        categorydf = emb_transformed[emb_transformed["label"] == cat]
        # print(categorydf)
        ax.scatter(
            categorydf[0],
            categorydf[1],
            # c=categorydf["label"].astype("category"),
            c = color,
            cmap="jet",
            alpha=alpha,
            label = categorydf["Dialogue Act"].unique()[0]
        )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")

    # Set legend outside of figure with enough space
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title("{} visualization of {} embeddings from {}".format(
    #         TSNE.__name__, plotname, dataset))
    fig.savefig(f"{dir}/TSNE_{plotname}.pdf",
                bbox_extra_artists=(lgd, ),
                bbox_inches='tight')
    if show:
        plt.show()

def model_sizes(setup):
    if setup == "A1":
        return ([1], [1], [32], 128)
    if setup == "A2":
        return ([1, 1], [1, 1], [32, 32], 128)
    if setup == "A3":
        return ([1, 1, 1], [1, 1, 1], [32, 32, 32], 128)

    if setup == "B1":
        return ([3], [3], [64], 128)
    if setup == "B2":
        return ([3, 3], [3, 3], [64, 64], 128)
    if setup == "B3":
        return ([3, 3, 3], [3, 3, 3], [64, 64, 64], 128)

    if setup == "C1":
        return ([5], [3], [128], 512)
    if setup == "C2":
        return ([5, 5], [3, 3], [128, 64], 512)
    if setup == "C3":
        return ([5, 5, 3], [3, 3, 1], [128, 64, 32], 512)

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

    parser.add_argument("--start_node",
                        help = "Are we including the start node?",
                        type = int,
                        default = 0)

    parser.add_argument("--aggregator",
                        help = "Which aggregation function to use?",
                        type = str,
                        choices = ["mean", "meanpool", "maxpool", "attention"])

    parser.add_argument("--model_size",
                        help = "Which model size use?",
                        type = str,
                        choices = ["A1", "A2", "A3",
                                   "B1", "B2", "B3",
                                   "C1", "C2", "C3"])

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
                                              args.num_test,
                                              args.start_node)

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
                                              args.num_test,
                                              args.start_node)

    else:
        print("Please enter a valid dataset: SwDA | MRDA | MANtIS | MSDialog | testing")
        sys.exit()

    # Report results and best model across
    fold_accuracies = []
    fold_microf1 = []
    fold_macrof1 = []
    fold_precision = []
    fold_losses = []
    # ------------------------------------

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
                                                                      full_graph,
                                                                      args.start_node)
        inference_node_indeces, inference_utterance_indeces = splitByDialogue(inference_split,
                                                                        full_graph,
                                                                        args.start_node)

        print(f"Split from fold {current_fold}:")
        print("Train on: ", train_utterance_indeces.shape, " nodes.")
        # Sample subgraph used to train/validate from the full graph
        graph_train_sampled = full_graph.subgraph(train_node_indeces)
        print(graph_train_sampled.info())

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
            print(train_multi_hot.shape)

        # -----------------------------------------------------
        # -- 4. Build GraphSAGE model and generator for train -
        # -----------------------------------------------------
        batch_size = args.batch_size
        # in_samples = [1] # <-- settings for A1
        # out_samples = [1]
        # layer_sizes = [32]
        # class_layer_size = 128
        in_samples, out_samples, layer_sizes, class_layer_size = model_sizes(args.model_size)

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
            layer_sizes=layer_sizes,
            aggregator = chooseAggregator(args.aggregator),
            generator=generator,
            bias=True,
            dropout=0.5,
        )
        # Define input and output tensors of graphsage model
        x_inp, x_out = graphsage_model.in_out_tensors()
        # Define the prediction layer
        if args.dataset in ['SwDA', 'MRDA']:
            first_layer = layers.Dense(units=class_layer_size, activation="relu")(x_out)
            prediction = layers.Dense(units=train_one_hot.shape[1], activation="softmax")(first_layer)
        else:
            first_layer = layers.Dense(units=class_layer_size, activation="relu")(x_out)
            prediction = layers.Dense(units=train_multi_hot.shape[1], activation="sigmoid")(first_layer)
        # -----------------------------------------------
        # -- 6. Train machine learning model ------------
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
        # -- 7. Perform testing on unseen dialogues -----
        # -----------------------------------------------

        generator = DirectedGraphSAGENodeGenerator(full_graph,
                                                   batch_size,
                                                   in_samples,
                                                   out_samples)

        if args.dataset in ['SwDA', 'MRDA']:
            inference_one_hot = target_encoding.transform(targets[inference_utterance_indeces])
            assert (len(inference_utterance_indeces) == len(inference_one_hot))
            inference_gen = generator.flow(inference_utterance_indeces, inference_one_hot, shuffle = True)

        else:
            inference_multi_hot = binarizeMultiLabelDataset(target_encoding, targets[inference_utterance_indeces])
            assert (len(inference_utterance_indeces) == len(inference_multi_hot))
            inference_gen = generator.flow(inference_utterance_indeces, inference_multi_hot, shuffle=True)

        # -----------------------------------------------
        # -- 8.Record testing metrics, f1micro, f1macro, precision, accuracy -----
        # -----------------------------------------------

        hold_out_loss_accuracy = model.evaluate(inference_gen)
        print(f"\nFold[{current_fold}] Hold Out Set Metrics:")
        print('Test loss:', hold_out_loss_accuracy[0])
        print('Test accuracy:', hold_out_loss_accuracy[1])
        fold_accuracies.append(hold_out_loss_accuracy[1])
        fold_losses.append(hold_out_loss_accuracy[0])

        hold_out_predictions = model.predict(inference_gen)

        if args.dataset in ['SwDA', 'MRDA']:
            y_pred_bool = np.argmax(hold_out_predictions, axis = 1)

            # Return y_test to single-digit labels
            y_test = np.argmax(inference_one_hot, axis = 1)
        else:
            # For binary crossentropy we use round to achieve a decision threshold at 0.5
            y_pred_bool = np.round(hold_out_predictions)
            y_test = inference_multi_hot

        f1_micro = f1_score(y_test, y_pred_bool, average = 'micro')
        f1_macro = f1_score(y_test, y_pred_bool, average='macro')
        precision = precision_score(y_test, y_pred_bool, average = 'micro')

        fold_microf1.append(f1_micro)
        fold_macrof1.append(f1_macro)
        fold_precision.append(precision)

        # Print model on the very last fold and save output
        # from final prediction as well as embeddings
        if current_fold == args.kfold:

            if args.start_node:
                dir = f'results/{args.expname}_{args.aggregator}_{args.dataset}_START'
            else:
		dir = f'results/{args.expname}_{args.aggregator}_{args.dataset}_NO-START'
	    if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

            # Save model summary in dir
            original_stdout = sys.stdout
            with open(f"{dir}/model_summary.txt", "w+") as f:
                sys.stdout = f
                print(model.summary())
                sys.stdout = original_stdout

            # Todo: Save final predicted output as dataframe in dir
            if args.dataset in ['SwDA', 'MRDA']:
                # Put prediction into one-hot-encoding using max
                y_pred = (hold_out_predictions == hold_out_predictions.max(axis = 1)\
                          [:, None]).astype(int)
                # transform labels back into textual (string) space
                y_pred = target_encoding.inverse_transform(y_pred)
                results = pd.Series(y_pred, index = inference_utterance_indeces)
                df = pd.DataFrame({"Predicted": results,
                                   "Ground Truth": targets[inference_utterance_indeces]})
                pickle.dump(df, open(f"{dir}/final_predictions.pickle", "wb"))

            else:
                y_pred = target_encoding.inverse_transform(y_pred_bool)
                results = pd.Series(y_pred, index = inference_utterance_indeces)
                df = pd.DataFrame({"Predicted": results,
                                   "Ground Truth": targets[inference_utterance_indeces]})
                pickle.dump(df, open(f"{dir}/final_predictions.pickle", "wb"))

            # Save embeddings from the GraphSAGE model for plotting in dir
            embedding_model = Model(inputs = x_inp, outputs = x_out)
            emb = embedding_model.predict(inference_gen)
            print(emb.shape)
            pickle.dump(emb, open(f"{dir}/graphEmbeddings.pickle", "wb"))

            # Save tSNE plot of embeddings and ground-truths
            plotTSNE(emb, pd.Series(targets[inference_utterance_indeces]),
                     dir, args.expname, show = False)

        current_fold += 1

    original_stdout = sys.stdout
    with open(f"{dir}/performance.txt", "w+") as f:
        sys.stdout = f

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(fold_accuracies)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {fold_losses[i]} - Accuracy: {fold_accuracies[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(fold_accuracies)} (+- {np.std(fold_accuracies)})')
        print(f'> Precision: {np.mean(fold_precision)} (+- {np.std(fold_precision)})')
        print(f'> F1-Micro: {np.mean(fold_microf1)} (+- {np.std(fold_microf1)})')
        print(f'> F1-Macro: {np.mean(fold_macrof1)} (+- {np.std(fold_macrof1)})')
        print(f'> Loss: {np.mean(fold_losses)}')
        print('-')


        sys.stdout = original_stdout



    original_stdout = sys.stdout
    with open(f"{dir}/parameters.txt", "w+") as f:
        sys.stdout = f

        print("Command line arguments:")
        for i in vars(args):
            print(i, ": ", getattr(args, i))

        print("GraphSAGE model parameters:")
        print("In samples:", in_samples)
        print("Out samples:", out_samples)
        print("Graph Layer sizes:", layer_sizes)
        print("Classification layer size:", class_layer_size)

        sys.stdout = original_stdout
