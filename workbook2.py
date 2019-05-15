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
        [0,1,0,0,1],
        [1,0,0,0,0],
        [1,0,0,1,1],
        [0,1,1,0,1],
        [0,0,1,1,0]
    ])
    labelsA = ['x1','x2','x3','x4','x5']

    B = np.array([
        [0,1,0],
        [1,0,1],
        [1,1,0]
    ])
    labelsB = ['y1','y2','y3']

    def sig(x):
        return np.tanh(x)
    def sig2(x):
        return 2*np.tanh(x)
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

    f = np.array([
        [zero,sig,zero,zero,sig],
        [sig,zero,zero,zero,zero],
        [sig,zero,zero,sig,sig],
        [zero,sig,sig,zero,sig],
        [zero,zero,sig,sig,zero]
    ])
    a = np.array([func1,func2,func3,func4,func5])

    # G = s.DirectedGraph(A, (a,f), labels=labelsA)
    # G.iterate(80, np.random.random(G.n)*100, graph=True, save_img=True, title="test2")

    # G.specialize(['x1','x2'])
    # G.iterate(80, np.random.random(G.n)*100, graph=True, save_img=True, title="test2 spec 1")

    h = np.array([
        [zero,func1,zero],
        [func1,zero,func1],
        [func1,func1,zero]
    ])
    a1 = np.array([zero,zero,zero])
    H = s.DirectedGraph(B, (a1,h), labels=labelsB)
    H.iterate(80, np.random.random(H.n)*10, graph=True, save_img=True, title='test_ex2')

    H.specialize(['y1','y2'])
    print(H.A)
    print(H.labeler)
    print(H.eigen_centrality())
    H.iterate(80, np.random.random(H.n)*10, graph=True, save_img=True, title='test_ex2 spec 1')
