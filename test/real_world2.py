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
from scipy.io import mmread
import pickle
import multiprocessing as mp

def export_colors(datafile, outfile):
    try:
        A = mmread(datafile)
        A = A.toarray()
        d = np.zeros_like(A)
        G = s.DirectedGraph(A, d)
        colors = G.coloring()
        with open(outfile, 'wb') as out:
            pickle.dump(colors, out)
    except Exception as e:
        with open('error_log.txt', 'w') as error_log:
            error_log.write(f'\n {e}')
    finally:
        return


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())

    # A = mmread('/home/ethan/Research/NetworkSpecialization/data/inf-power/inf-power.mtx')
    files = [
        ('/home/ethan/Research/NetworkSpecialization/data/socfb-American75/socfb-American75.mtx', 'fb_colors')
        ('/home/ethan/Research/NetworkSpecialization/data/inf-road_usa/inf-road_usa.mtx','us_road_colors'),
    ]
        # ('/home/ethan/Research/NetworkSpecialization/data/bn-mouse_visual-cortex_1/bn-mouse_visual-cortex_1.edges','mouse_viz_colors'),
        # ('/home/ethan/Research/NetworkSpecialization/data/rt_gop/rt_gop.edges','gop_rt_path'),
        # ('/home/ethan/Research/NetworkSpecialization/data/rt_obama/rt_obama.edges','obama_rt_colors'),
        # ('/home/ethan/Research/NetworkSpecialization/data/rt-pol/rt-pol.txt','rt_pol_colors')



    results = [pool.apply(export_colors, args=(files[i][0], files[i][1])) for i in range(len(files))]

    pool.close()
