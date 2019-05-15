import numpy as np
import scipy.linalg as la

# Problems 1-2
class PageRank:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
        labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        n (int): number of nodes
    """

    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate A_hat. Save A_hat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        # find the number of nodes we are dealing with and copy the adjacency matrix
        n = A.shape[0]
        A_hat = A.copy().astype(np.float64)
        # if the labels are nonetype then we will assign them to be integer values
        if type(labels) == type(None):
            labels = np.arange(n)
        elif len(labels) != n:
            raise ValueError('The number of labels is not equal to the number of columns in the matrix')
        elif type(labels) != np.ndarray:
            labels = np.array(labels)
        # for each column in A_hat we check that it is
        for i in range(n):
            s = np.sum(A_hat[:,i])
            if s == 0:
                A_hat[:,i] = 1/n
            elif s != 1:
                A_hat[:,i] /= s
        # store the attributes that we need
        self.A = A_hat
        self.labels = labels
        self.n = n

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Use a system of linear equations to rank the nodes
        I = np.eye(self.n)
        p = la.solve(I - epsilon*self.A, (1-epsilon)/self.n*np.ones(self.n))
        rank = {self.labels[i]:p[i] for i in range(self.n)}
        return rank

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # us the eigenvalues to rank the nodes
        B = (epsilon*self.A + ((1-epsilon)/self.n) * np.ones_like(self.A))
        eigs = la.eig(B)
        i = np.argmax(eigs[0])
        p = eigs[1][:,i]
        p /= p.sum()
        rank = {self.labels[i]:p[i] for i in range(self.n)}
        return rank

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # use an iterative method to rank the nodes
        one = np.ones(self.n)
        p0 = one.copy() / self.n
        for _ in range(maxiter):
            p = epsilon*self.A @ p0 + ((1-epsilon)/self.n)*one
            if la.norm(p - p0, 1) < tol:
                break
            p0 = p
        rank = {self.labels[i]:p[i] for i in range(self.n)}
        return rank
