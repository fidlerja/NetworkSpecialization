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
    A = np.array([
        [0,1,1,1],
        [1,0,0,1],
        [1,1,0,1],
        [0,1,1,0]
    ])
    labelsA = ['x1','x2','x3','x4']

    def sig(x):
        return anp.tanh(x)
    def sig2(x):
        return -1.*anp.tanh(x)
    def sig3(x):
        return -2.0*anp.tanh(x)
    def zero(x):
        return 0*x
    def f1(x):
        return 0.9*x + 1/5
    def f2(x):
        return 0.9*x + 7/5
    def f3(x):
        return 0.9*x + 0.5

    f = np.array([
        [zero,zero,zero,sig],
        [sig,zero,sig,zero],
        [sig,sig,zero,zero],
        [zero,sig,sig,zero]
    ])
    a = np.array([f3,f1,f2,f3])
    G = s.DirectedGraph(A, (a,f), labels=labelsA)
    G.iterate(80, np.random.random(G.n), graph=True, save_img=True, title="../graphs/test_simple")

    G.specialize(['x1','x4'])
    G.iterate(80, np.random.random(G.n), graph=True, save_img=True, title="../graphs/test_simple_spec1")
    print(G.eigen_centrality())

    for i in range(3):
        base = np.random.choice(G.indices, 2)
        G.specialize(base)
    print(G.n)
    G.iterate_with_perturbations(300, np.random.random(G.n), ([50], 50), graph=True, save_img=True, title='big network with perturbation')