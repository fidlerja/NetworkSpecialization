import specializer as s
import ranker as r
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time
import autograd.numpy as anp
import sys
np.set_printoptions(threshold=sys.maxsize)

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
        return anp.sin(x)
    def f2(x):
        return 0.9*x + 7/5
    def f3(x):
        return 0.9*x + 0.5

    f = np.array([
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig2,sig2,sig2,sig2,sig2,sig2,sig2,sig2],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig2,sig2,sig2,sig2,sig2,sig2,sig2,sig2],
        [sig,sig2,sig,sig,sig,sig,sig2,sig]
    ])

    a = np.array([zero,f1,f1,f1,f1,f1,f1,f1])
    labels = ['1','2','3','4','5','6','7','8']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # print(G.coloring())
    # G.network_vis()#use_eqp=True)
    # G.iterate(20,np.random.random(8),graph=True)
    base = ['1','8','5','6','7']
    G.specialize(base)
