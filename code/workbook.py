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
<<<<<<< HEAD
<<<<<<< HEAD
    # A = np.array([
    #     [0,0,0,0,0,0,1],
    #     [1,0,1,1,0,0,0],
    #     [0,1,0,0,0,0,0],
    #     [0,0,1,0,0,0,0],
    #     [0,0,1,0,0,0,0],
    #     [0,0,0,1,1,0,1],
    #     [0,0,0,1,0,1,0]
    # ])
    # labels = ["a", "b", "c", 'd','e','f','g']

    def sig(x):
        # print(x)
=======

    def sig(x):
>>>>>>> 46a72d2d85226a042751738f6291551b95c7238b
=======

    def sig(x):
>>>>>>> 66ed2b073ae9b08576d8f2a880d94cf11402f785
        return anp.tanh(x)
    def sig2(x):
        y = -2.0*anp.tanh(x)
        return y
    def zero(x):
        return 0*x
<<<<<<< HEAD
<<<<<<< HEAD
    def func1(x):
        # print(x)
        y = 9/10*x + 7/4
        # print(y)
=======
=======
>>>>>>> 66ed2b073ae9b08576d8f2a880d94cf11402f785
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
<<<<<<< HEAD
>>>>>>> 46a72d2d85226a042751738f6291551b95c7238b
=======
>>>>>>> 66ed2b073ae9b08576d8f2a880d94cf11402f785
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
<<<<<<< HEAD
<<<<<<< HEAD
    a = np.array([func1, func2, func3])
    labels = ['x1', 'x2', 'x3']

    # after the first specialization
    # C = np.array([[0,0,0,1],[0,0,0,1],[0,1,0,0],[1,0,1,0]])
    # labels2 = ['y1', 'y2', 'y3', 'y4']
    # b = np.array([func1, func1, func2, func3])
    # g = np.array([[zero, zero, zero, sig2],
    #                 [zero, zero, zero, sig2],
    #                 [zero, sig, zero, zero],
    #                 [sig, zero, sig, zero]])


    # H = s.DirectedGraph(C, (b,g), labels=labels2)
    # H.iterate(80, np.random.random(H.n), graph=True, save_img=True, title='first spec hard code corrected')


    G = s.DirectedGraph(B, (a,f), labels=labels)
    print(G.eigen_centrality(), '\n')
    print(G.detect_sync(), '\n')
    # G.iterate_with_perterbations(1000, np.random.random(G.n), ([500], 150), graph=True, save_img=False, title="3 nodes")

    G.specialize(['x1', 'x2'], verbose=False)
    # G.iterate_with_perterbations(1000, np.random.random(G.n), ([500], 150), graph=True, save_img=False, title="3 nodes")
    G.iterate(80, np.random.random(G.n), graph=True, save_img=True, title='../graphs/spec on x1, x2')
    print(G.eigen_centrality(), '\n')
    print(G.detect_sync(), '\n')

    G.specialize(['x2', 'x3.1', 'x3.2'], verbose=False)
    # G.iterate_with_perterbations(1000, np.random.random(G.n), ([500], 150), graph=True, save_img=False, title="3 nodes")
    G.iterate(80, np.random.random(G.n)*10, graph=True, save_img=True, title='../graphs/spec on x2, x31, x32')
    print(G.eigen_centrality(), '\n')
    print(G.detect_sync(), '\n')

    # N = nx.DiGraph(G.A.T)
    # nx.draw_networkx_labels(N, pos=nx.spring_layout(N), labels=G.labeler)
    # nx.draw(N, pos=nx.spring_layout(N))
    # plt.draw()
    # plt.show()
=======
=======
>>>>>>> 66ed2b073ae9b08576d8f2a880d94cf11402f785
    a = np.array([chaotic_func1, chaotic_func2, chaotic_func3])
    labels = ['x1', 'x2', 'x3']

    G = s.DirectedGraph(B, (a,f), labels=labels)

    for i in range(3):
        base = np.random.choice(G.indices, 2, replace=False)
        G.specialize(base)
    print(G.n)
    G.iterate_with_perturbations(300, np.random.random(G.n), ([150], 10), graph=True, save_img=True, title='../graphs/chaotic cgnn with perturbation')
<<<<<<< HEAD
    print(G.spectral_radius())
>>>>>>> 46a72d2d85226a042751738f6291551b95c7238b
=======
    print(G.spectral_radius())
>>>>>>> 66ed2b073ae9b08576d8f2a880d94cf11402f785
