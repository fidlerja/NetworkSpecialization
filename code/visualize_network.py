# visualize_network.py
"""Network Specialization Research
Erik Hannesson and Ethan Walker
16 May 2019
"""
from ds import detect_sync
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph



def network_vis(G, iter_matrix=False, title="Network Visualization", save_img=False):
    """ Creates a visualization of the network G
    Parameters:
        G (DirectedGraph): A DirectedGraph object to visualize
        iter_matrix (ndarray): allows you to explicitly pass in the dynamics matrix of G
            If left as False, then we run the dynamics simulation here
        title (str): Title of visualization
        save_img (bool): if True, saves the visualization with filename 'filename'
        filename (str): filename
    """
    # find synchronized communities and create a networkx graph object
    communities = detect_sync(G, iter_matrix=iter_matrix)
    nxG = nx.relabel.relabel_nodes(nx.DiGraph(G.A), G.labeler)

    nx.r
