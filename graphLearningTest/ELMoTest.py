"""
Super simple, try to embed every utterance of the SwDA .csv file 'SwDA.csv'.
Start by embedding the first few and the last few lines.
Check the content of their vectors.
Then save the embeddings.
Then load them.
Check that the vectors are still the same.
"""

import pandas as pd
import numpy  as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import sys

print("To create ELMo embeddings takes a very long time, do not attempt " + \
      "without CPU!, Use this script locally only to load pre-generated ELMo embeddings")

pd.set_option('display.max_colwidth', 200)

dataset = pd.read_csv("../CSVData/SwDA.csv")
print(dataset.head())

# sys.exit()

pd.options.mode.chained_assignment = None

# Do some preprocessing,
# Fix apostrophies that are spaced apart
dataset.loc[:, 'proc_utterance'] = \
    dataset['Utterance'].apply(lambda x: re.sub(' \'','\'', str( re.sub(' n\'t','n\'t', str(x)) )))

# remove exclamation and questionmarks
punctuation = '!?'
dataset.loc[:, 'proc_utterance'] = dataset['proc_utterance'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

pd.options.mode.chained_assignment = 'warn'

dataset = dataset[['proc_utterance', 'Dialogue Act', 'Dialogue ID']]

print("Loading elmo from tensorhub")
tf.compat.v1.disable_eager_execution()
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

# Function for extracting the mean ELMo vector for each utterance
def elmo_vectors(x):
    print(x)
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
    # embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.compat.v1.reduce_mean(embeddings,1))


# ------- Load ELMo embeddings as completed on gdrive
pickle_in = open("../finished_ELMo/swda_ELMo/elmo_swda_id[0-20].pickle", "rb")
swda_0to20 = pickle.load(pickle_in)
print(swda_0to20[27])
print(dataset[0:1]["proc_utterance"])
first_embed = elmo_vectors(dataset[0:1]["proc_utterance"])