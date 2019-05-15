import specializer as s
import ranker as r
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    reload(s)
    A = np.array([
        [0,0,0,1],
        [1,0,1,0],
        [1,1,0,0],
        [0,1,1,0]
    ])
    labelsA = ['x1','x2','x3','x4']
    def sig(x):
        return np.tanh(x)
    def zero(x):
        return 0*x
    f = np.array([
        [zero,zero,zero,sig],
        [sig,zero,sig,zero],
        [sig,sig,zero,zero],
        [zero,sig,sig,zero]
    ])
    a = np.array([zero,zero,zero,zero])
    G = s.DirectedGraph(A, (a,f), labels=labelsA)
    G.iterate(80, np.random.random(G.n)*100, graph=True, save_img=True, title="test_simple")

    G.specialize(['x1','x4'])
    G.iterate(80, np.random.random(G.n)*100, graph=True, save_img=True, title="test_simple_spec1")
