"""
File used from tutorial page from the StellarGraph library.
https://stellargraph.readthedocs.io/en/stable/demos/node-classification/graphsage-inductive-node-classification.html
"""

import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import stellargraph as sg
from stellargraph import globalvar
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt

# Load the dataset
dataset = datasets.PubMedDiabetes()
graph_full, labels = dataset.load()
print(graph_full.info())

# We use the label pandas Series to index which nodes we are working on from the full graph,
# therefore we never have to touch the full graph, just slice and split the ID-label Series.
# Labels for training : lables_sampled
labels_sampled = labels.sample(frac = 0.8, replace = False, random_state = 101)

# Extract subgraph for training
graph_sampled = graph_full.subgraph(labels_sampled.index)

# Can be useful if we can get Gephi to work on Ubuntu
# Gnx = graph_sampled.to_networkx(feature_attr=None)
# nx.write_graphml(Gnx, "nameOfVisual.graphml")

# Tricky part: SPLITS
# First extract training data as 5% of the sampled subgraph,
# use remaining for validation
train_labels, val_labels = model_selection.train_test_split(
    labels_sampled,
    train_size=0.05,
    test_size=None,
    stratify=labels_sampled,
    random_state=42,
)

# Turn labels into one-hot encodings
target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_labels)
val_targets = target_encoding.transform(val_labels)

# Create a node generator for undirected graph
batch_size = 50
num_samples = [10, 10]
generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)

# create iterator for training data
train_gen = generator.flow(train_labels.index, train_targets, shuffle=True)

# Make graphsage model
graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
)
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
# Make generator for validation data
val_gen = generator.flow(val_labels.index, val_targets)

# Fit the model
history = model.fit(
    train_gen, epochs=15, validation_data=val_gen, verbose=1, shuffle=False
)
sg.utils.plot_history(history, return_figure = True)
# plt.show()


# ---- INDUCTIVE PART -----
# We need a new generator so that we can generate random walks over the whole
# graph (with the held out part re-inserted)
generator = GraphSAGENodeGenerator(graph_full, batch_size, num_samples)

# Receive the labels for the nodes that were excluded from
# the random walks.
hold_out_nodes = labels.index.difference(labels_sampled.index)
print(hold_out_nodes)
labels_hold_out = labels[hold_out_nodes]
print(labels_hold_out)
hold_out_targets = target_encoding.transform(labels_hold_out)
hold_out_gen = generator.flow(hold_out_nodes, hold_out_targets)

# Make predictions for the held out set
hold_out_predictions = model.predict(hold_out_gen)

# Look at the held out predictions

hold_out_predictions = target_encoding.inverse_transform(hold_out_predictions)
results = pd.Series(hold_out_predictions, index=hold_out_nodes)
df = pd.DataFrame({"Predicted": results, "True": labels_hold_out})
print(df.head(10))

hold_out_metrics = model.evaluate(hold_out_gen)
print("\nHold Out Set Metrics:")
for name, val in zip(model.metrics_names, hold_out_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Make TSNE plot of the held out node embeddings
embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(hold_out_gen)
X = emb
y = np.argmax(target_encoding.transform(labels_hold_out), axis=1)
if X.shape[1] > 2:
    transform = TSNE  # PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=hold_out_nodes)
    emb_transformed["label"] = y
else:
    emb_transformed = pd.DataFrame(X, index=hold_out_nodes)
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = y

alpha = 0.7

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GraphSAGE embeddings of hold out nodes for pubmed dataset".format(
        transform.__name__
    )
)
plt.show()