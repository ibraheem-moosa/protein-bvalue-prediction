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


class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self._Xes = []
        self._yes = []
        for xf,yf in files:
            X = torch.from_numpy(scsp.load_npz(xf).toarray()).reshape((-1, 21))
            y = torch.from_numpy(np.load(yf)['y']).reshape((-1, 1))
            X = X[10:-10]
            y = y[10:-10]
            assert(X.shape[0] == y.shape[0])
            self._Xes.append(X)
            self._yes.append(y)
        
    def __getitem__(self, idx):
        X = self._Xes[idx]
        y = self._yes[idx]
        return X.view(1, X.shape[0], X.shape[1]), y.view(1, y.shape[0], y.shape[1])

    def __len__(self):
        return len(self._Xes)

    def get_max_min(self):
        max_val = -10000
        min_val = 100000
        for i in range(len(self)):
            max_val = max(max_val, self._yes[i].max().item())
            min_val = min(min_val, self._yes[i].min().item())
        return max_val, min_val

    def get_avg_median(self):
        med = 0.0
        for i in range(len(self)):
            med += self._yes[i].median().item()
        return med / len(self)

    def plot_max_dist(self):
        max_vals = [self._yes[i].max().item() for i in range(len(self))]
        plt.hist(max_vals)
        plt.show()

    def plot_min_dist(self):
        min_vals = [self._yes[i].min().item() for i in range(len(self))]
        plt.hist(min_vals)
        plt.show()

    def plot_med_dist(self):
        med_vals = [self._yes[i].median().item() for i in range(len(self))]
        plt.hist(med_vals)
        plt.show()

    def plot_whole_dist(self):
        bvals = np.concatenate([self._yes[i].numpy() for i in range(len(self))], axis=0)
        plt.hist(bvals, bins=100)
        plt.show()

    def plot_whole_var_dist(self):
        bvals = np.concatenate([self._yes[i].numpy() for i in range(len(self))], axis=0)
        plt.hist(bvals, weights=np.abs(bvals), bins=100)
        plt.show()


