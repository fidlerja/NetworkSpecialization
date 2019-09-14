# test_coloring.py
import specializer as s
import numpy as np
from importlib import reload



if __name__ == "__main__":
    reload(s)
    def zero(x):
        return 0
    
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
G.network_vis(use_eqp=True,spec_layout=False)
print(G.coloring())
G.specialize(['1','4'])
G.network_vis(use_eqp=True,spec_layout=False)
print(G.coloring())
