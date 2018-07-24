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
            assert(X.shape[0] == y.shape[0])
            self._Xes.append(X)
            self._yes.append(y)
        '''
        self._Xes = sorted(self._Xes, key=lambda x: x.shape[0])
        self._yes = sorted(self._yes, key=lambda y: y.shape[0])
        to_be_collated_Xes = [self._Xes[0]]
        to_be_collated_yes = [self._yes[0]]
        collated_Xes = []
        collated_yes = []
        for X, y in zip(self._Xes, self._yes):
            assert(X.shape[0] == y.shape[0])
            if X.shape[0] == to_be_collated_Xes[-1].shape[0]:
                assert(y.shape[0] == to_be_collated_yes[-1].shape[0])
                to_be_collated_Xes.append(X)
                to_be_collated_yes.append(y)
            else:
                collated_Xes.append(torch.stack(to_be_collated_Xes))
                collated_yes.append(torch.stack(to_be_collated_yes))
                to_be_collated_Xes = [X]
                to_be_collated_yes = [y]
        self._Xes = collated_Xes
        self._yes = collated_yes
        '''

    def __getitem__(self, idx):
        X = self._Xes[idx]
        y = self._yes[idx]
        return X.view(1, X.shape[0], X.shape[1]), y.view(1, y.shape[0], y.shape[1])

    def __len__(self):
        return len(self._Xes)


