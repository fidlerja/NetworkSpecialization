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
