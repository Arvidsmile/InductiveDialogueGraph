"""
1.  Choose a train-split and use it as vocabulary to
    embed all the utterances into TF-IDF vectors.
2.  Create two graphs, one (TV)train/valid-graph and one for testing
    For edges, create directed edges between all the nodes in each conversation.

3.  The valid-part of the TV-graph has concealed node-labels that
    are not propagating error during training. Only the "train-nodes"
    can propagate error.
4.  Train the model by sending the signal into a softmax classifier in the
    end,
5.  Perform testing on the held out test-graph.


"""