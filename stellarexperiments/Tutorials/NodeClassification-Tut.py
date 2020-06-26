"""
File used from tutorial page from the StellarGraph library.
https://stellargraph.readthedocs.io/en/stable/demos/node-classification/graphsage-node-classification.html
"""

import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanAggregator, \
    MeanPoolingAggregator, MaxPoolingAggregator, AttentionalAggregator

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Loading CORA network
dataset = datasets.Cora()
G, node_subjects = dataset.load()
print(G.info())
# node_subjects are the labels in a pandas dataframe
print(set(node_subjects))

# Use scikit learn to split data into train/test. Here we
# Only split on labels. Not the graph itself. We use stratify
# node_subjects to make sure that we have the same class proportion in
# the samples as in the whole, non-split dataset
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.1, test_size=None, stratify=node_subjects
)

print(Counter(train_subjects))
print(Counter(test_subjects))

# Convert data to numeric one-hot encoded arrays
target_encoding = preprocessing.LabelBinarizer()
# ..we use the word target for ground-truth label
train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)

print(train_targets.shape)
print(test_targets.shape)

# Create graphSAGENodeGenerator object which feeds data from a graph
# to a model. It requires batch_size and the number of nodes to sample
# in a decided number of layers.
batch_size = 50
num_samples = [10, 5] #two layers

# Create data generator for our graph, specified by which type
# of model (GraphSAGE) and the learning task (Node) ...
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

print(train_subjects.index)

# Create an iterator for our training data, this takes the indeces of the
# nodes in the graph to be used for training, as well as their respective
# one-hot encoded label vectors
train_gen = generator.flow(train_subjects.index, train_targets, shuffle = True)

# Specify the graph-learning model

graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
    aggregator = MeanAggregator
)

# Extract the input and output tensors of the model. Set predictions
# of the model to be a softmax layer taking output tensor as its input.
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
model = Model(inputs = x_inp, outputs = prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
# To validate/test we need another generator for testing data, no shuffle needed
test_gen = generator.flow(test_subjects.index, test_targets)
history = model.fit(
    train_gen,
    epochs = 20,
    validation_data = test_gen,
    verbose = 2,
    shuffle = False
)

# Plot training/validation accuracy/loss
print(sg.utils.plot_history(history, return_figure = True))
plt.savefig("testsave.png")
# plt.show()

# Test run model on the testing generator (again :) )
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Make predictions using the model
all_nodes = node_subjects.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)

# Use inverse_transform of the LabelBinarizer to turn the
# one-hot encodings back into their textual labels
node_predictions = target_encoding.inverse_transform(all_predictions)
df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
print(df.head(10))


# Generate the embeddings without doing class prediction and
# plot them as TSNE embeddings, by using x_out we get the output
# tensor from our graphSAGE model which has already been trained.
embedding_model = Model(inputs = x_inp, outputs = x_out)
emb = embedding_model.predict(all_mapper)
print(emb.shape)

X = emb
# We want the labels now in integer format..
y = np.argmax(target_encoding.transform(node_subjects), axis=1)

if X.shape[1] > 2:
    transform = TSNE  # PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_subjects.index)
    emb_transformed["label"] = y
else:
    emb_transformed = pd.DataFrame(X, index=node_subjects.index)
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = y

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GraphSAGE embeddings for cora dataset".format(transform.__name__)
)
plt.show()