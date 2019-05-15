# ds.py
"""
Network Specialization Research
Erik Hannesson and Ethan Walker
15 May 2019
"""
import specializer
import numpy as np

def detect_sync(G, iter_matrix=False, iters=80, sync_tol=1e-2, otol=1e-1):
    """
    Determines which nodes in a network synchronize.

    Parameters:
        G (DirectedGraph): The DirectedGraph of interest

        iter_matrix (ndarray): mxn array containing the values
            of m dynamic iterations of the n nodes of G

        iters (int): number of iterations to produce iter_matrix (for use when iter_matrix is not explicitly passed in)

        sync_tol (float): tolerance for synchronization

        otol (float): tolerance for stability

    Returns:
        sync_communities (tuple): of the form ((community), bool), where the bool represents if the community is stable
    """
    if iter_matrix is False:
        iter_matrix = G.iterate(iters, np.random.random(G.n)*10)


    # number of nodes in network
    n = iter_matrix.shape[1]

    # only check the last few rows for convergence
    A = iter_matrix[-5:, :]

    # tracks which nodes/communities have synchronized
    sync_communities = []
    sync_nodes = []

    for node in range(n):
        if node in sync_nodes:
            break
        else:
            # sync_ind is a list of nodes synchronized with 'node'
            sync_ind = list(np.where(np.all(np.isclose((A.T - A[:, node]).T, 0, atol=sync_tol), axis=0))[0])
            # track which nodes have already synchronized so we don't double check
            sync_nodes.extend(sync_ind)
            # track communities of synchronization
            sync_communities.append((tuple(map(lambda x: G.labeler[x], sync_ind)), np.all(np.isclose(A[:, node] - A[-1, node], 0, atol=otol))))

    return sync_communities
