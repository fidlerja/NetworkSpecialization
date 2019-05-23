# visualize_network.py
"""Network Specialization Research
Erik Hannesson and Ethan Walker
16 May 2019
"""
import specializer as s
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def network_vis(G, iter_matrix=False, spec_layout=False, lin=False, lin_dyn=None, title="Network Visualization", save_img=False, filename='network_viz'):
    """ Creates a visualization of the network G
    Parameters:
        G (DirectedGraph): A DirectedGraph object to visualize
        iter_matrix (ndarray): allows you to explicitly pass in the dynamics matrix of G
            If left as False, then we run the dynamics simulation here
        spec_layout (bool): if False will display with random layout, otherwise it will use
            the spectral layout option
        lin (bool): Adds edge labels if the dynamics are linear
        lin_dyn (ndarray): the linear dynamics array
        title (str): Title of visualization
        save_img (bool): if True, saves the visualization with filename 'filename'
        filename (str): filename
    """
    # find synchronized communities
    communities = G.detect_sync(iters=80)

    # create a dictionary mapping each node to its community
    group_dict = {}
    for i in range(len(communities)):
        for node in communities[i][0]:
            group_dict[node] = i

    # create (and relabel) a networkx graph object
    nxG = nx.relabel.relabel_nodes(nx.DiGraph(G.A.T), G.labeler)

    # set community membership as an attribute of nxG
    nx.set_node_attributes(nxG, group_dict, name='community')

    # list of community number in order of how the nodes are stored
    colors = [group_dict[node] for node in nxG.nodes()]

    plt.figure()

    if lin:
        # for display of edge dynamics (linear dynamics only)
        edge_weights = nx.get_edge_attributes(nxG, 'weight')
        # edit the edge weights to be the correct dynamics
        for edge in edge_weights:
            i, j = G.origination(G.indexer[edge[1]]), G.origination(G.indexer[edge[0]])
            edge_weights[edge] = lin_dyn[i, j]



    if spec_layout:
        # draw the network
        nx.draw_networkx(nxG, pos=nx.drawing.spectral_layout(nxG), node_size=1000,
                         arrowsize=20, node_color=colors, cmap=plt.cm.Set3)
        if lin:
            # add edge weights
            nx.draw_networkx_edge_labels(nxG, nx.drawing.spectral_layout(nxG),
                                         edge_labels=edge_weights)

    else:
        # draw the network
        nx.draw_networkx(nxG, pos=nx.drawing.layout.planar_layout(nxG), node_size=1000,
                         arrowsize=20, node_color=colors, cmap=plt.cm.Set3)
        if lin:
            # add edge weights
            nx.draw_networkx_edge_labels(nxG, nx.drawing.layout.planar_layout(nxG),
                                         edge_labels=edge_weights)

    plt.title(title)
    if save_img:
        plt.savefig(filename)

    plt.show()
