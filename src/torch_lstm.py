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

    if len(sys.argv) < 4:
        print('Usage: python3 regression.py data_dir protein_list checkpoint_dir [warm_start_model] [warm_start_epoch]')
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
    
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    #indices = indices[:100]
    train_indices = indices[:int(0.8 * len(indices))]
    validation_indices = indices[int(0.8 * len(indices)):]

    if len(sys.argv) == 6:
        warm_start_model_params = torch.load(sys.argv[4])
        warm_start_last_epoch = int(sys.argv[5])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 1
    init_lr = 2.0 ** -7
    momentum = 0.9
    weight_decay = 1e-4
    gamma = 0.9
    hidden_size = 16
    hidden_scale = 1.0
    num_hidden_layers = 1
    output_layer_depth = 2
    ff_scale = 0.6
    grad_clip = 10.0
    nesterov = True

    print('Initial LR: {}'.format(init_lr))
    #print('Momentum: {}'.format(momentum))
    print('Weight Decay: {}'.format(weight_decay))
    print('Gradient Clip: {}'.format(grad_clip))
    print('Hidden Scale: {}'.format(hidden_scale))
    print('FF Scale: {}'.format(ff_scale))
    print('Hidden Layer Size: {}'.format(hidden_size))
    print('Output Layer Depth: {}'.format(output_layer_depth))
    print('Number of Hidden Layers: {}'.format(num_hidden_layers))
    #print('Nesterov: {}'.format(nesterov))
    print('Adam')
    #print('Adadelta')

    net = LSTMNeuralNetwork(21,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_layer_depth=output_layer_depth,
            hidden_scale=hidden_scale, ff_scale=ff_scale, 
            init_lr=init_lr, gamma=gamma, weight_decay=weight_decay)
    
    net.train(dataset, train_indices, validation_indices, patience=20, model_dir=sys.argv[3], warm_start_last_epoch=warm_start_last_epoch,
            warm_start_model_params=warm_start_model_params)
    #train_nn(net, dataset, train_indices, validation_indices, sys.argv[3])
    #scores = cross_validation(net, dataset, indices, 10, 0.40)
    #print(scores)
    '''
    param_grid = {'init_lr' : 2.0 ** np.arange(-7, -6), 'hidden_size' : [4,8,16],
                    'weight_decay' : 10.0 ** np.arange(-4,-3), 'gamma' : [0.9],
                    'output_layer_depth' : [2],
                    'num_hidden_layers' : [1]}
    def set_init_lr(net, value):
        net.init_lr = value
    def set_hidden_size(net, value):
        net.reset_hidden_size(value)
    def set_weight_decay(net, weight_decay):
        net.reset_weight_decay(weight_decay)
    def set_gamma(net, gamma):
        net.reset_gamma(gamma)
    def set_output_layer_depth(net, output_layer_depth):
        net.reset_output_layer_depth(output_layer_depth)
    def set_num_hidden_layers(net, num_hidden_layers):
        net.reset_num_hidden_layers(num_hidden_layers)

    param_set_funcs = {'init_lr' : set_init_lr, 'hidden_size' : set_hidden_size, 
                        'weight_decay' : set_weight_decay, 'gamma' : set_gamma, 
                        'output_layer_depth' : set_output_layer_depth,
                        'num_hidden_layers' : set_num_hidden_layers}

    results = gridsearchcv(net, dataset, indices, 3, 0.45, param_grid, param_set_funcs)
    for i in range(len(results)):
        for param_name, param_value in results[i][0].items():
            print('{}:{}'.format(param_name, param_value), end=' ')
        print('score: {0:.4f}'.format(results[i][1]))
    #print(results)
    '''