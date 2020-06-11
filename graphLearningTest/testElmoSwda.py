"Test downloaded ELMo embeddings using a logistic regression classifier"


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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# 1. Load all the ELMo embeddings in order:
pickle_in = open(f"../finished_ELMo/swda_ELMo/elmo_swda_id[0-20].pickle", "rb")
swda_ELMo = pickle.load(pickle_in)
for index in tqdm(range(20, 2220, 20)):
    pickle_in = open(f"../finished_ELMo/swda_ELMo/elmo_swda_id[{index}-{index + 20}].pickle", "rb")
    swda_ELMo = np.concatenate((swda_ELMo, pickle.load(pickle_in)), axis = 0)

# 2. Pre-process the swda dataset in order:
def processSwDA(dataset):
  pd.options.mode.chained_assignment = None
  # Do some preprocessing,
  # Fix apostrophies that are spaced apart
  dataset.loc[:, 'proc_utterance'] = \
      dataset['Utterance'].apply(lambda x: re.sub(' \'','\'', str( re.sub(' n\'t','n\'t', str(x)) )))

  # remove exclamation and questionmarks
  punctuation = '!?'
  dataset.loc[:, 'proc_utterance'] = dataset['proc_utterance'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

  # lowercase the text
  # dataset.loc[:, 'proc_utterance'] = dataset['proc_utterance'].str.lower()

  pd.options.mode.chained_assignment = 'warn'

  dataset = dataset[['proc_utterance', 'Dialogue Act']]
  return dataset

swda_processed = processSwDA(pd.read_csv("../CSVData/swda.csv"))

# 3. Build a logistic regression classifier on the ELMo embeddings


print(swda_processed["Dialogue Act"].value_counts())
swda_processed["Dialogue Act"] = pd.factorize(swda_processed["Dialogue Act"])[0]
print(swda_processed["Dialogue Act"].value_counts())

xtrain, xvalid, ytrain, yvalid = train_test_split(swda_ELMo,
                                                  swda_processed['Dialogue Act'],
                                                  random_state=42,
                                                  test_size=0.2)



lreg = LogisticRegression(max_iter = 10000, solver = "lbfgs", multi_class = "auto", verbose = True)
print("xtrain.shape: ", xtrain.shape,
      "ytrain.shape:", ytrain.shape,
      "xvalid.shape:", xvalid.shape,
      "yvalid.shape:", yvalid.shape)

scores = cross_val_score(lreg, xtrain[0:50000], ytrain[0:50000], cv = 10, verbose = True)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

preds_valid = lreg.predict(xvalid)
print(f1_score(yvalid, preds_valid, average = "micro"))
print(f1_score(yvalid, preds_valid, average = "macro"))
print(f1_score(yvalid, preds_valid, average = "weighted"))



