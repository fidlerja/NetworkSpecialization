import sys
sys.path.append('/home/ethan/Research/NetworkSpecialization/core')
from specializer import *
from scipy.io import mmread
from importlib import reload
import pickle


if __name__ == "__main__":
    # A = mmread('/home/ethan/Research/NetworkSpecialization/data/inf-power/inf-power.mtx')
    A = mmread('../data/inf-power/inf-power.mtx')
    A = A.toarray()
    d = np.zeros_like(A)
    G = DirectedGraph(A, d)
    colors = G.coloring()
    with open('colors', 'wb') as out:
        pickle.dump(colors, out)


