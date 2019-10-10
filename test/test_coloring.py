# test_coloring.py
import sys
import specializer_pyvis as s
import numpy as np
from importlib import reload



# if __name__ == "__main__":
reload(s)
def zero(x):
    return 0

def random_matrix(n):
    A = np.random.randint(0,2,(n,n))
    A *= (np.eye(n) == 0)

    f = [[zero for _ in range(n)] for _ in range(n)]
    f = np.array(f)
    #print(f)

    a = np.diag(f)

    return (A, f, a)

def random_test(M):
    (A,f,a) = M
    labels = [str(i) for i in range(1,len(A)+1)]
    #print(A)
    #print(labels)
    G = s.DirectedGraph(A, (a,f), labels=labels)
    # G.network_vis(use_eqp=True,spec_layout=True)
    # Q0 = G.colors
    # G.specialize(['1','4'])
    # G.network_vis(use_eqp=True,spec_layout=True)
    # Q1 = G.colors
    print(G.coloring())
    print(G.checker())

    # return G, Q0, Q1

def basic_test():
    A = np.array([[0,0,0,1],
                  [1,0,1,0],
                  [1,1,0,0],
                  [0,1,1,0]])

    f = np.array([[zero,zero,zero,zero],
                  [zero,zero,zero,zero],
                  [zero,zero,zero,zero],
                  [zero,zero,zero,zero]])

    a = np.diag(f)



    labels = ['1','2','3','4']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # G.network_vis(use_eqp=True,spec_layout=False)
    # # print(G.coloring())
    # G.specialize(['1','4'])
    # G.network_vis(use_eqp=True,spec_layout=False)
    # print(G.coloring())
    print(G.coloring())
    print(G.checker())
