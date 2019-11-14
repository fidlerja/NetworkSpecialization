import sys
import os
path = os.getcwd()
sys.path.insert(1, path[:-4])

import core.specializer as s
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time
import autograd.numpy as anp
import pickle
from scipy.io import mmread

if __name__ == "__main__":
    # A = mmread('/home/ethan/Research/NetworkSpecialization/data/inf-power/inf-power.mtx')
    A = mmread('../data/inf-power/inf-power.mtx')
    A = A.toarray()
    d = np.zeros_like(A)
    G = s.DirectedGraph(A, d)
    colors = G.coloring()
    with open('colors', 'wb') as out:
        pickle.dump(colors, out)


