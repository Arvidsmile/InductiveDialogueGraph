"""
Requires folder CSVData in parent-folder containing
the datasets converted into csv tabular format with
columns ["Actor", "Dialogue Act", "Dialogue ID", "Utterance"].
Uses ELMo embeddings of size 1024 by averaging the embeddings
of each token in a given input utterance.

"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import re
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
os.environ['KMP_WARNINGS'] = '0'

# (OLD) Function for generating elmo embeddings from a dataframe column
# containing utterances. Output : A numpy matrix of size (num_rows, 1024)
# Input: x, a column of size num_rows from chosen dataset
# def elmo_vectors(x, elmo):
#     embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
#
#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())
#         sess.run(tf.compat.v1.tables_initializer())
#         # return average of ELMo features
#         output_embeddings = sess.run(tf.reduce_mean(embeddings,1))
#
#     return output_embeddings
#

def elmo_vectors2(session, tf_output, tf_input, x):
    result = session.run(tf_output, feed_dict={tf_input: x.tolist()})
    return result

# Function for plotting ELMo embeddings in 2-dimensions
# using TSNE.
# Input: embeddings = matrix of size num_vectors * 1024,
#       DAlabels = column vector if size num_vectors * 1
#       dataset = "string with dataset name for saving",
#       show = False, if True, calls plot.show(), if not we only
#       save the image as a .pdf
def plotTSNE(embeddings, DAlabels, dataset = "testing", method = "ELMo", show = False):
    print("Plot the tsne-representation of the embeddings")
    trans = TSNE(n_components=2)

    # Create TSNE embeddings
    emb_transformed = pd.DataFrame(trans.fit_transform(embeddings))

    integerLabels = pd.factorize(DAlabels)[0]
    emb_transformed["label"] = integerLabels
    emb_transformed["Dialogue Act"] = DAlabels

    # Plot a scatterplot for each dialogue act category
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(10, 10))
    label_colors = iter(cm.rainbow(np.linspace(0, 1, len(emb_transformed["label"].unique()))))
    for cat, color in zip(emb_transformed["label"].unique(), label_colors):
        categorydf = emb_transformed[emb_transformed["label"] == cat]
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
    plt.title("{} visualization of {} embeddings from {}".format(
            TSNE.__name__, method, dataset))
    fig.savefig(f"../output_images/{method}_TSNE_{dataset}.pdf",
                bbox_extra_artists=(lgd, ),
                bbox_inches='tight')
    if show:
        plt.show()

# Pre-processing functions: ------
# Takes as argument a loaded pandas dataframe where each row contains an utterance text.
# Performs preprocessing specific to the dataset and returns the updated dataframe
# containing a new column "proc_utterance"
def preprocessSwDA(dataset):
    pd.options.mode.chained_assignment = None
    # Do some preprocessing,
    # Fix apostrophies that are spaced apart
    dataset.loc[:, 'proc_utterance'] = \
        dataset['Utterance'].apply(lambda x: re.sub(' \'','\'', str( re.sub(' n\'t','n\'t', str(x)) )))

    # remove any exclamation and question marks
    punctuation = '!?'
    dataset.loc[:, 'proc_utterance'] = dataset['proc_utterance'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    # lowercase the text
    dataset.loc[:, 'proc_utterance'] = dataset['proc_utterance'].str.lower()

    pd.options.mode.chained_assignment = 'warn'
    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    # If we have empty locations, replace them with dashes to avoid Nan Values in embeddings
    dataset.at[empty_locations, "proc_utterance"] = "---"
    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    return dataset

def preprocessMANtIS(dataset):
    pd.options.mode.chained_assignment = None
    # 1. Remove anything within (and including)  < ... >
    dataset.loc[:, 'proc_utterance'] = \
        dataset['Utterance'].apply(lambda x: re.sub('[<].*?[>]', '', str(x)))

    # 2. Remove all URL's
    dataset.loc[:, 'proc_utterance'] = \
        dataset['proc_utterance'].apply(
            lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                             'LINK',
                             str(x),
                             flags=re.MULTILINE))

    pd.options.mode.chained_assignment = 'warn'

    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    # If we have empty locations, replace them with dashes to avoid Nan Values in embeddings
    dataset.at[empty_locations, "proc_utterance"] = "---"
    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    return dataset

def preprocessMSDialog(dataset):
    pd.options.mode.chained_assignment = None
    # 1. Remove anything within (and including)  < ... >
    dataset.loc[:, 'proc_utterance'] = \
        dataset['Utterance'].apply(lambda x: re.sub('[<].*?[>]', '', str(x)))

    # 2. Remove all URL's
    dataset.loc[:, 'proc_utterance'] = \
        dataset['proc_utterance'].apply(
            lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                             'LINK',
                             str(x),
                             flags=re.MULTILINE))

    pd.options.mode.chained_assignment = 'warn'

    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    # If we have empty locations, replace them with dashes to avoid Nan Values in embeddings
    dataset.at[empty_locations, "proc_utterance"] = "---"
    empty_locations = np.where(dataset['proc_utterance'].apply(lambda x: x == ''))[0].tolist()
    print(f"Empty strings at {empty_locations}")
    return dataset

def preprocessMRDA(dataframe):
    pass

# Function that performs the work of creating embeddings.
# Embeds the whole dataset with the set step_size (number of
# utterances in each batch), creates a TSNE-plot of the embeddings
# as well as saving the embeddings in a pickle file.
def pipeline(dataset, proc_func, step_size = 10, testing = False):
    df = pd.read_csv(f"../CSVData/{dataset}.csv")
    # 1. Perform preprocessing of corpus
    df = proc_func(df)

    # 2. Create computational graph for generating ELMo embeddings
    print("Loading ELMo Module from tensorhub and creating comp. graph...")
    g = tf.Graph()
    with g.as_default():
        # We will be feeding 1D tensors of text into the graph.
        tf_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        embedded_text = elmo(tf_input, signature="default", as_dict=True)["elmo"]
        tf_output = tf.reduce_mean(embedded_text, 1)
        init_op = tf.group([tf.compat.v1.global_variables_initializer(),
                            tf.compat.v1.tables_initializer()])
    g.finalize()
    print("Done loading ELMo module.")

    # Create session and initialize.
    session = tf.compat.v1.Session(graph=g)
    session.run(init_op)

    # 3. Generate the embeddings
    print(f"Creating ELMo embeddings {step_size} utterances at a time.")
    batches = [df[i:i + step_size] for i in range(0, df.shape[0], step_size)]
    if testing:
        print("Generating only 30 embeddings.")
        embeddings = [elmo_vectors2(session,
                                    tf_output,
                                    tf_input,
                                    batch['proc_utterance']) for batch in tqdm(batches[0:30])]
    else:
        print("Generating all embeddings.")
        embeddings = [elmo_vectors2(session,
                                    tf_output,
                                    tf_input,
                                    batch['proc_utterance']) for batch in tqdm(batches)]

    embeddings = np.concatenate(embeddings, axis=0)

    print("Completed generating ELMo embeddings.")
    if testing:
        filepath = f"../ELMoPickled/ELMo_{args.dataset}_testing.pickle"
    else:
        filepath = f"../ELMoPickled/ELMo_{args.dataset}.pickle"

    print(f"Saving as pickle: {filepath}")
    pickle_out = open(filepath, "wb")
    pickle.dump(embeddings, pickle_out)
    pickle_out.close()

    if testing:
        plotTSNE(embeddings,
                 df["Dialogue Act"][0:embeddings.shape[0]],
                 dataset=args.dataset,
                 method="ELMo-testing",
                 show=False)
    else:
        plotTSNE(embeddings,
                 df["Dialogue Act"][0:embeddings.shape[0]],
                 dataset=args.dataset,
                 method="ELMo",
                 show=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help = "Which dataset do you want to generate ELMo embeddings for?",
                        type = str,
                        default = "SwDA")
    parser.add_argument("--testing",
                        help = "Are we in testing mode?",
                        type = int,
                        default = 0)
    parser.add_argument("--step_size",
                        help = "Size of each batch of utterances used in ELMo",
                        type = int,
                        default = 10)
    args = parser.parse_args()

    if args.dataset == "SwDA":
        # Note: On a gpu-cluster with large memory access, step_size should be
        # set to a larger number to generate embeddings more efficiently.
        # testing should be set to True in order to generate only (30 * step_size)
        # amount of embeddings, for testing purposes.
        pipeline(args.dataset, preprocessSwDA,
                 step_size = args.step_size, testing = args.testing)

    elif args.dataset == "MANtIS":
        pipeline(args.dataset, preprocessMANtIS,
                 step_size = args.step_size, testing = args.testing)

    elif args.dataset == "MSDialog":
        pipeline(args.dataset, preprocessMSDialog,
                 step_size = args.step_size, testing = args.testing)


    elif args.dataset == "MRDA":
        pass

    elif args.dataset == "testing":
        print("Testing the generation of a dummy utterance")
        dummydf = pd.DataFrame([
                    {"Utterance": "Hello, how are you doing?",
                     "Dialogue Act" : "Greeting"},
                    {"Utterance": "Hello, I am doing fine.",
                     "Dialogue Act": "Greeting"},
                    {"Utterance": "Do you like apples?",
                     "Dialogue Act": "Yes-No-Question"},
                    {"Utterance": "Yes, I like apples",
                     "Dialogue Act": "Answer"},
                    {"Utterance": "Do you enjoy movies?",
                     "Dialogue Act": "Yes-No-Question"},
                    {"Utterance": "Yes, I enjoy movies.",
                     "Dialogue Act": "Answer"},
                    {"Utterance": "I think oranges are tasty.",
                     "Dialogue Act": "Statement"},
                    {"Utterance": "I think oranges are stupid.",
                     "Dialogue Act": "Statement"},
                    {"Utterance": "The cat is happy.",
                     "Dialogue Act": "Statement"},
                    ])

        print("Loading ELMo Module from tensorhub and creating comp. graph...")
        g = tf.Graph()
        with g.as_default():
            # We will be feeding 1D tensors of text into the graph.
            tf_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
            embedded_text = elmo(tf_input, signature="default", as_dict=True)["elmo"]
            tf_output = tf.reduce_mean(embedded_text, 1)
            init_op = tf.group([tf.compat.v1.global_variables_initializer(),
                                tf.compat.v1.tables_initializer()])
        g.finalize()
        print("Done loading ELMo module.")
        # Create session and initialize.
        session = tf.compat.v1.Session(graph=g)
        session.run(init_op)

        embeddings = elmo_vectors2(session, tf_output, tf_input, dummydf["Utterance"])
        plotTSNE(embeddings,
                 dummydf["Dialogue Act"],
                 dataset = args.dataset,
                 method = "ELMo",
                 show = False)

    else:
        print("Please enter a valid dataset: SwDA | MRDA | MANtIS | MSDialog | testing")
