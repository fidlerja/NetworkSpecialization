import specializer as s
import ranker as r
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    reload(s)
    A = np.array([[0,0,0,0,0,0,1],[1,0,1,1,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0]
    ,[0,0,1,0,0,0,0],[0,0,0,1,1,0,1],[0,0,0,1,0,1,0]])
    labels = ["a", "b", "c", 'd','e','f','g']
    # G = s.DirectedGraph(A, labels)

    # G.specialize_graph(["a",'e'], verbose=True)
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

    # basic cohen - grossberg neural network
    B = np.array([[0,0,1],[1,0,0],[1,1,0]])
    f = np.array([[zero, zero, sig2],
                    [ sig, zero, zero],
                    [ sig, sig, zero]])
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

    """
test stuff
    def set_func(a,f):
        complex_F = [None]*3
        for i in range(3):
            asdf = create(B,f,i)
            complex_F[i] = asdf
        return complex_F
    def create(B,f,i):
        return lambda x: sum(B[i,j] * f[i,j](x[j]) for j in range(3))

    complex_F = set_func(a,f)

    g = lambda x: [ complex_F[k](x) for k in range(3)]

    print(g([1,1,1]))

    for i in range(3):
        print(complex_F[i]([1,1,1]))
    """

    # H = s.DirectedGraph(C, (b,g), labels=labels2)
    # H.iterate(80, np.random.random(H.n), graph=True, save_img=True, title='first spec hard code corrected')


    G = s.DirectedGraph(B, (a,f), labels=labels)
    # G.iterate(80, np.random.random(G.n), graph=True, save_img=True, title="3 nodes")
    s = time.time()
    x = G.iterate(800, [1,1,1], graph=False, save_img=True, title="CGNN")
    e = time.time()
    print(e-s)
    # print(G.eigen_centrality())

    G.specialize(['x1', 'x2'], verbose=False)
    G.iterate(80, np.random.random(G.n), graph=False, save_img=True, title='spec on x1, x2')
    # print(G.eigen_centrality())

    print(G.A)
    print(G.labeler)
    G.specialize(['x2', 'x3.1', 'x3.2'], verbose=False)
    G.iterate(80, np.random.random(G.n)*10, graph=True, save_img=True, title='spec on x2, x31, x32')
    # print(G.eigen_centrality())

    # G.specialize(['x1.1', 'x1.2', 'x2'], verbose=False)
    # G.iterate(80, np.random.random(G.n)*10, graph=False, save_img=True, title='spec on x11, x12, x2')
    # print(G.eigen_centrality())

    # N = nx.DiGraph(G.A.T)
    # nx.draw_networkx_labels(N, pos=nx.spring_layout(N), labels=G.labeler)
    # nx.draw(N, pos=nx.spring_layout(N))
    # plt.draw()
    # plt.show()
