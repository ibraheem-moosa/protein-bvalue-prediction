import torch
from torch import nn
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect
import matplotlib.pyplot as plt
from math import sqrt
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import sys
import time
import os
import random
from torch_rnn_dataset import *

def write_preds(proteins, preds, dirname):
    for i in range(len(proteins)):
        protein = proteins[i]
        with open(os.path.join(dirname, protein), "w") as f:
            for j in range(len(preds[i])):
                f.write('{}\n'.format(preds[i][j]))


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

def get_avg_pcc(y_true, y_pred, lengths):
    pcc =[]
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    for i in range(len(lengths)):
        l = lengths[i]
        yt = y_true[i].flatten()
        yp = y_pred[i].flatten()
        pcc.append(pearsonr(yt[:l], yp[:l])[0])

    pcc = np.array(pcc)
    pcc[np.isnan(pcc)] = 0
    return np.mean(pcc)

def get_avg_mse(y_true, y_pred, lengths):
    mse =[]
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    for i in range(len(lengths)):
        l = lengths[i]
        yt = y_true[i].flatten()
        yp = y_pred[i].flatten()
        mse.append(mean_squared_error(yt[:l], yp[:l]))

    mse = np.array(mse)
    mse[np.isnan(mse)] = 0
    return np.mean(mse)


def cross_validation(net, files, batch_size, k, threshold):
    n = len(files) // k
    r = len(files) - n * k
    fold_lengths = [n + 1] * r + [n] * (k - r)
    cumulative_fl = [0]
    for fl in fold_lengths:
        cumulative_fl.append(cumulative_fl[-1] + fl)
    scores = []
    for i in range(k):
        print('Cross Validation Fold: {}'.format(i))
        train_files = []
        validation_files = []
        for j in range(k):
            if j == i:
                validation_files.extend(files[cumulative_fl[j]:cumulative_fl[j+1]])
            else:
                train_files.extend(files[cumulative_fl[j]:cumulative_fl[j+1]])
        train_dataset = ProteinDataset(train_files, batch_size)
        validation_dataset = ProteinDataset(validation_files, batch_size)
        train_mses, validation_mses = net.train(train_dataset, validation_dataset) 
        validation_mse = min(validation_mses)
        scores.append(validation_mse)
        print(validation_mse)
        print(threshold)
        print(validation_mse > threshold)
        if validation_mse > threshold:
            break
    return scores


def get_param_config(param_grid, keys):
    if len(keys) == 0:
        yield None
    else:
        for value in param_grid[keys[0]]:
            for rest_config in get_param_config(param_grid, keys[1:]):
                yield keys[0], value, rest_config

def gridsearchcv(net, data_files, batch_size, k, threshold, param_grid, param_set_funcs):
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
        scores = cross_validation(net, data_files, batch_size, k, threshold)
        mean_score = sum(scores) / len(scores)
        print('Got score {} for params {}'.format(mean_score, param_config_dict))
        result.append((param_config_dict, mean_score))
    return result


