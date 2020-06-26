"""
File used from tutorial page from the StellarGraph library.
https://stellargraph.readthedocs.io/en/stable/demos/node-classification/directed-graphsage-node-classification.html
"""

import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import DirectedGraphSAGENodeGenerator
from stellargraph.layer import DirectedGraphSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt

# Load the dataset
dataset = datasets.Cora()
G, node_subjects = dataset.load(directed=True)

print(node_subjects.index)

# Split the data, using labels
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.1, test_size=None, stratify=node_subjects
)

# Use a label binarizer to convert results into one hot
target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)


# For directed graph, we keep track of sampling from nodes coming
# in and nodes coming out (for our random walk)
batch_size = 50
in_samples = [5, 2]
out_samples = [5, 2]
generator = DirectedGraphSAGENodeGenerator(G,
                                           batch_size,
                                           in_samples,
                                           out_samples)

# make training iterator
train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)
graphsage_model = DirectedGraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=False, dropout=0.5,
)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
test_gen = generator.flow(test_subjects.index, test_targets)
history = model.fit(
    train_gen, epochs=5, validation_data=test_gen, verbose=2, shuffle=False
)