import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect
import matplotlib.pyplot as plt
from math import sqrt
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import sys
import time
import os
import random
from torch_helper import *
from torch_rnn import *
from torch_rnn_dataset import *

if __name__ == '__main__':

    torch.set_printoptions(precision=2, linewidth=140)
    torch.manual_seed(42)

    if len(sys.argv) < 3:
        print('Usage: python3 regression.py data_dir protein_list')
        exit()

    print(time.strftime('%Y-%m-%d %H:%M'))

    with open(sys.argv[2]) as protein_list_file:
        protein_list = protein_list_file.read().split()
        protein_list = [s.upper().strip() for s in protein_list]

    X_files = []
    y_files = []

    for protein in protein_list:
        X_files.append(os.path.join(sys.argv[1], 'X_' + protein + '_rnn_.npz'))
        y_files.append(os.path.join(sys.argv[1], 'y_' + protein + '_rnn_.npz'))

    files = list(zip(X_files, y_files))
    dataset = ProteinDataset(files)
    print('Dataset init done ', len(dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))
    print(dataset.get_max_min())
    print(dataset.get_avg_median())
    #dataset.plot_max_dist()
    #dataset.plot_min_dist()
    #dataset.plot_med_dist()
    dataset.plot_whole_dist()
    dataset.plot_whole_var_dist()
