# test_ds.py
import specializer as s
import ranker as r
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
from ds import detect_sync


if __name__ == "__main__":
    reload(s)

    """FUNCTIONS"""
    def sig(x):
        # print(x)
        return np.tanh(x)
    def sig2(x):
        return -2 * np.tanh(x)
    def zero(x):
        return 0
    def func1(x):
        # print(x)
        y = 9/10 * x + 7/4
        # print(y)
        return y
    def func2(x):
        # print(x)
        y = 9/10 * x + 5/4
        # print(y)
        return y
    def func3(x):
        # print(x)
        y = 9/10 * x + 2/4
        # print(y)
        return y


    B = np.array([[0,0,1],[1,0,0],[1,1,0]])
    f = np.array([[zero, zero, sig2],
                    [ sig, zero, zero],
                    [ sig, sig, zero]])
    a = np.array([func1, func2, func3])
    labels = ['x1', 'x2', 'x3']

    G = s.DirectedGraph(B, (a,f), labels=labels)
    print(detect_sync(G, G.iterate(80, np.random.random(G.n)*5, graph=True)))

    G.specialize(['x2', 'x3'], verbose=False)
    print(detect_sync(G, G.iterate(80, np.random.random(G.n)*5, graph=True)))
