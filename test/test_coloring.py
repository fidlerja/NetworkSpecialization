# test_coloring.py
import sys
sys.path.insert(0,'/home/adam/Desktop/Research/NetworkSpecialization/core/')
import specializer as s
import numpy as np
from importlib import reload



if __name__ == "__main__":
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

def random_test(M,count):
    (A,f,a) = M
    labels = [str(i) for i in range(1,len(A)+1)]
    #print(A)
    #print(labels)
    G = s.DirectedGraph(A, (a,f), labels=labels)
    G.coloring()
    Q0 = G.colors
    # print('Specializing graph {}...'.format(count))
    # G.specialize(['1','4'])
    # Q1 = G.colors
    # print('Done!')
    bool = G.sum_checker()
    # print(bool)
    if bool == False:
        try:
            G.network_vis(use_eqp=True)#,silent=False)
        except:
            pass
        print(G.colors)
        print(G.A)
        print('Graph {} is colored incorrectly!'.format(count))
        return G

    #G.network_vis(use_eqp=True,spec_layout=True)
    # Q1 = G.colors

    return G#, Q0, Q1

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
    # G.network_vis(use_eqp=True)
    print(G.colors)
    return G

    G.specialize(['1','4'],recolor=True)
    # G.network_vis(use_eqp=True)
    print(G.colors)
    return G

def problem_test():
    A = np.array([[0, 1, 1, 1, 1, 0],
                 [1, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1 ,0 ,1],
                 [0, 0, 1, 0, 1, 0]]
)
    f = np.array([[zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero]])
    a = np.diag(f)
    labels = ['1','2','3','4','5','6']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # G.network_vis(use_eqp=True)
    G.coloring()
    # print(G.colors)
    return G

def triv_test():
    A = np.array([[0,1,1,1,1,1],
                  [0,0,1,1,0,0],
                  [1,1,0,1,0,0],
                  [1,0,1,0,0,1],
                  [1,0,0,0,0,1],
                  [0,0,0,1,0,0]])
    f = np.array([[zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero],
                  [zero,zero,zero,zero,zero,zero]])
    a = np.diag(f)
    labels = ['1','2','3','4','5','6']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # G.network_vis(use_eqp=True)
    G.coloring()
    print(G.colors)
    return G
