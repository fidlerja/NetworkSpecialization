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
        y = -2.0*anp.tanh(x)
        return y
    def zero(x):
        return 0*x
    def func1(x):
        y = 9/10*x + 7/4
    def chaotic_func1(x):
        y = anp.tanh(4*x*(1-x)) + 1.75
        return y
    def chaotic_func2(x):
        y = anp.tanh(4*x*(1-x)) + 1.25
        return y
    def chaotic_func3(x):
        y = anp.tanh(4*x*(1-x)) + 0.25
        return y
    def func1(x):
        y = (9/10)*x + 1.75
        return y
    def func2(x):
        y = (9/10)*x + 1.25
        return y
    def func3(x):
        y = (9/10)*x + 0.5
        return y

    # basic cohen-grossberg neural network
    B = np.array([
        [0,0,1],
        [1,0,0],
        [1,1,0]
    ])
    f = np.array([
        [zero, zero, sig2],
        [ sig, zero, zero],
        [ sig, sig, zero]
    ])
    a = np.array([chaotic_func1, chaotic_func2, chaotic_func3])
    labels = ['x1', 'x2', 'x3']

    G = s.DirectedGraph(B, (a,f), labels=labels)

    for i in range(3):
        base = np.random.choice(G.indices, 2, replace=False)
        G.specialize(base)
    print(G.n)
    G.iterate_with_perturbations(300, np.random.random(G.n), ([150], 10), graph=True, save_img=True, title='../graphs/chaotic cgnn with perturbation')