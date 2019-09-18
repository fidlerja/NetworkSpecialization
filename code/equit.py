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
        [0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [1,1,0,1,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [1,0,0,0,0,1,0,0],
        [1,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,0,0],
        [0,1,1,1,1,1,1,0]
    ])

    def sig(x):
        return anp.tanh(x)
    def sig2(x):
        return -1.*anp.tanh(x)
    def sig3(x):
        return -2.0*anp.tanh(x)
    def zero(x):
        return 0*x
    def f1(x):
        return 0.5*x + 1/5
    def f2(x):
        return 0.9*x + 7/5
    def f3(x):
        return 0.9*x + 0.5

    f = np.array([
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1],
        [f1,f1,f1,f1,f1,f1,f1,f1]
    ])

    a = np.array([sig,sig,sig,sig,sig,sig,sig,sig])
    labels = ['1','2','3','4','5','6','7','8']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # print(G.coloring())
    # G.network_vis()#use_eqp=True)
    G.iterate(300,np.random.random(8),graph=True)
    base = ['1','8']
    G.specialize(base)
    G.iterate(300,np.random.random(38),graph=True)
    # G.network_vis()#use_eqp=True)
    # print(G.n)