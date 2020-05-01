# InductiveDialogueGraph
Originally cloned from https://github.com/williamleif/GraphSAGE.git/ by the 
authors of "Inductive Representation Learning on Large Graphs" 2018 
(https://arxiv.org/abs/1706.02216), this repo contains the code of a
 project using GraphsSAGE in order to perform inductive node 
 classification as a graphical approach to dialogue act classification (DAC). 
 
 #### To run graph creation
Navigate to root folder 'InductiveDialogueGraph'
 ```shell script
arvid@arvid-ThinkPad-E580:~/Desktop/Project_thesis/InductiveDialogueGraph$
python -m graphCreation.main --dataset [mantis, mrda, msdialog, swda]
    --type path-graph unique-utterance actor-nodes
    --split "0.8 0.1 0.1"
    --graph_name "default_graph_name"
    --random_seed 1337
```

`graphCreation.main` module will create a graph using the `networkx` library 
and store three graphs for train, validation and test, respectively. A useful 
visualization of the three graphs are stored as interactive `.html` files in the 
same output folder to easily see the structure of the resulting graphs. Output 
folder and files are named based on `--graph_name` argument.

#### To run GraphSAGE
Begin by creating the docker image required by running
```shell script
/bin/bash build_docker_image.sh
```
The run the docker by running:
```shell script
/bin/bash run_docker_image.sh
```

Follow instructions inside GraphSAGE folder. For example, run the provided
toy-example by doing the following
```shell script
cd GraphSAGE
./example_unsupervided.sh
```