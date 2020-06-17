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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
os.environ['KMP_WARNINGS'] = '0'

def plotTSNE(embeddings, DAlabels, dataset = "testing", method = "ELMo", show = False):
    print("Plot the tsne-representation of the embeddings")
    trans = TSNE(n_components=2, verbose = True)

    # Create TSNE embeddings
    emb_transformed = pd.DataFrame(trans.fit_transform(embeddings))

    integerLabels = pd.factorize(DAlabels)[0]
    emb_transformed["label"] = integerLabels
    emb_transformed["Dialogue Act"] = DAlabels

    # Plot a scatterplot for each dialogue act category
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(10, 10))
    label_colors = iter(cm.rainbow(np.linspace(0, 1, len(emb_transformed["label"].unique()))))
    for cat, color in tqdm(zip(emb_transformed["label"].unique(), label_colors)):
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
    fig.savefig(f"../../output_images/{method}_TSNE_{dataset}_pop.pdf",
                bbox_extra_artists=(lgd, ),
                bbox_inches='tight')
    if show:
        plt.show()

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

# Create graph and finalize (finalizing optional but recommended).
# Run this code once -------------------------------
#g = tf.Graph()
#with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
#  tf_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
#  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable = False)
#  embedded_text = elmo(tf_input, signature="default", as_dict=True)["elmo"]
#  tf_output = tf.reduce_mean(embedded_text,1)
#  init_op = tf.group([tf.compat.v1.global_variables_initializer(),
#                      tf.compat.v1.tables_initializer()])
#g.finalize()

# Create session and initialize.
#session = tf.compat.v1.Session(graph=g)
#session.run(init_op)
#-------------------------------------------------------

# ---- inference time!
# result = session.run(tf_output, feed_dict={tf_input: ["My first sentence",
#                                                      "My second sentence",
#                                                      "Ice cream is tasty!"]})

# ---- or as a function (that can be put into a loop using the batches trick)
#def elmo_vectors2(session, tf_output, tf_input, x):
#  result = session.run(tf_output, feed_dict={tf_input: x.tolist()})
#  return result


if __name__ == '__main__':
    print("loading swda.csv")
    swda = pd.read_csv("../../CSVData/SwDA.csv")
    print("loading embeddings")
    embedding_file = open("../../ELMoPickled/ELMo_SwDA.pickle", 'rb')
    ELMo = pickle.load(embedding_file)

#    print("preprocessing swda")
#    swda_proc = preprocessSwDA(swda)
    print("plotting TSNE")
    plotTSNE(ELMo, swda["Dialogue Act"])

    swda_proc = preprocessSwDA(swda)

    pca = PCA(n_components = 100)

    pca_result = pca.fit_transform(ELMo)
    print(ELMo.shape)
    print(pca_result.shape)

    plotTSNE(pca_result, swda_proc["Dialogue Act"])
