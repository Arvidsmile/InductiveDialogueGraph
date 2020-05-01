
import os
from pyvis.network import Network

"""
Author: Arvid L. 2020, UvA

The class GraphVisualizer uses pyvis
to create an interactive graph visualization that
can be viewed and manipulated in a regular web-browser.
Graph .html files are saved in their output folder.
Creates one .html visualization for the train, valid and 
test graphs respectively. 
"""

## todo:
## remove this, you can use os.chdir('path here') to move
## to desired folder before saving the html file.

class GraphVisualizer(object):

    def __init__(self, graph_obj, graph_folder, graph_file):
        print("Visualizer created!")
        net = Network()