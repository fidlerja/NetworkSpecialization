import specializer as s
import ranker as r
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time
import autograd.numpy as anp

if __name__ == "__main__":
    reload(s)


    def sig(x):
        return anp.tanh(x)
    def sig2(x):
        return 2*anp.tanh(x)
    def zero(x):
        return 0*x
    def func1(x):
        return 0.5*x
    def func2(x):
        return 0.25*x
    def func3(x):
        return 0.5*x
    def func4(x):
        return 0.5*x
    def func5(x):
        return 0.5*x

    A = np.array([
        [0,0,1,1],
        [1,0,0,0],
        [0,1,0,0],
        [0,1,0,0]
    ])
    labels = ['x1','x2','x3','x4']
    f = np.array([
        [zero,zero,zero,zero],
        [zero,zero,zero,zero],
        [zero,zero,zero,zero],
        [zero,zero,zero,zero]
    ])
    a = np.array([zero,zero,zero,zero])
    G = s.DirectedGraph(A, (a, f), labels)
    G.specialize(['x1'], verbose=True)
    G.network_vis()