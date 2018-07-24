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


def summarize_tensor(tensor):
    return torch.max(tensor).item(), torch.min(tensor).item(), torch.mean(tensor).item(), torch.std(tensor).item()

def close_event():
    plt.close()

def plot_true_and_prediction(y_true, y_pred):
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=10000)
    timer.add_callback(close_event)
    plt.title('Bidirectional, 8 Hidden States, 2 Output Layers')
    plt.plot(y_pred, 'y-')
    plt.plot(y_true, 'g-')
    timer.start()
    plt.show()

def summarize_nn(net):
    print('##############################################################')
    for name, param in net.named_parameters():
        print('---------------------------------------------------------------')
        print(name)
        print(summarize_tensor(param))
    print('##############################################################')

def get_avg_pcc(net, dataset, indices):
    pcc = []
    for i in indices:
        x, y = dataset[i]
        y_pred = net.predict(x)
        for j in range(x.shape[0]):
            pcc.append(pearsonr(y_pred.numpy()[j].flatten(), y.numpy()[j].flatten())[0])

    pcc = np.array(pcc)
    pcc[np.isnan(pcc)] = 0
    return np.mean(pcc)

def cross_validation(net, dataset, indices, k, threshold):
    n = len(indices) // k
    r = len(indices) - n * k
    fold_lengths = [n + 1] * r + [n] * (k - r)
    cumulative_fl = [0]
    for fl in fold_lengths:
        cumulative_fl.append(cumulative_fl[-1] + fl)
    scores = []
    for i in range(k):
        print('Cross Validation Fold: {}'.format(i))
        train_indices = []
        validation_indices = []
        for j in range(k):
            if j == i:
                validation_indices.extend(indices[cumulative_fl[j]:cumulative_fl[j+1]])
            else:
                train_indices.extend(indices[cumulative_fl[j]:cumulative_fl[j+1]])
        train_pccs, validation_pccs = net.train(dataset, train_indices, validation_indices) 
        validation_pcc = max(validation_pccs)
        scores.append(validation_pcc)
        if validation_pcc < threshold:
            break
    return scores


def get_param_config(param_grid, keys):
    if len(keys) == 0:
        yield None
    else:
        for value in param_grid[keys[0]]:
            for rest_config in get_param_config(param_grid, keys[1:]):
                yield keys[0], value, rest_config

def gridsearchcv(net, dataset, indices, k, threshold, param_grid, param_set_funcs):
    result = []
    num_of_params = len(param_grid)
    for param_config in get_param_config(param_grid, list(param_grid.keys())):
        next_param_config = param_config
        param_config_dict = dict()
        while True:
            key, value, next_param_config = next_param_config
            param_config_dict[key] = value
            param_set_funcs[key](net, value)
            if next_param_config is None:
                break
            
        print('Running CV for params {}'.format(param_config_dict))
        scores = cross_validation(net, dataset, indices, k, threshold)
        mean_score = sum(scores) / len(scores)
        print('Got score {} for params {}'.format(mean_score, param_config_dict))
        result.append((param_config_dict, mean_score))
    return result


