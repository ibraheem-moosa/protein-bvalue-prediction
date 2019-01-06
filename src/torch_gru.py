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

    if len(sys.argv) < 5:
        print('Usage: python3 regression.py data_dir protein_list checkpoint_dir pred_dir [warm_start_model] [warm_start_epoch]')
        exit()

    if len(sys.argv) == 7:
        warm_start_params = torch.load(sys.argv[5])
        warm_start_last_epoch = int(sys.argv[6])
    else:
        warm_start_params = None
        warm_start_last_epoch = -1


    pred_dir = sys.argv[4]
    print(time.strftime('%Y-%m-%d %H:%M'))

    with open(sys.argv[2]) as protein_list_file:
        protein_list = protein_list_file.read().split()
        protein_list = [s.upper().strip() for s in protein_list]
    
    indices = list(range(len(protein_list)))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(indices))]
    val_indices = train_indices[int(0.9 * len(train_indices)):]
    train_indices = train_indices[:int(0.9 * len(train_indices))]
    test_indices = indices[int(0.8 * len(indices)):]

    X_files = []
    y_files = []

    for protein in protein_list:
        X_files.append(os.path.join(sys.argv[1], 'X_' + protein + '_rnn_.npz'))
        y_files.append(os.path.join(sys.argv[1], 'y_' + protein + '_rnn_.npz'))

    files = list(zip(X_files, y_files))
    train_files = [files[i] for i in train_indices]

    batch_size = 1
    
    dataset = ProteinDataset([files[i] for i in train_indices], batch_size, [protein_list[i] for i in train_indices])
    print('Dataset init done ', len(dataset))
    val_dataset = ProteinDataset([files[i] for i in val_indices], batch_size,[protein_list[i] for i in val_indices])
    print('Val Dataset init done ', len(val_dataset))
    test_dataset = ProteinDataset([files[i] for i in test_indices], 1,[protein_list[i] for i in test_indices])
    print('Test Dataset init done ', len(test_dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))
    
    init_lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    gamma = 0.9
    hidden_size = 16
    hidden_scale = 1.0
    num_hidden_layers = 2
    output_layer_depth = 2
    ff_scale = 0.6
    grad_clip = 10.0
    max_iter = 100
    patience = 5
    nesterov = True

    print('Initial LR: {}'.format(init_lr))
    print('Batch Size: {}'.format(batch_size))
    #print('Momentum: {}'.format(momentum))
    print('Weight Decay: {}'.format(weight_decay))
    print('Gamma: {}'.format(gamma))
    print('Gradient Clip: {}'.format(grad_clip))
    print('Hidden Scale: {}'.format(hidden_scale))
    print('FF Scale: {}'.format(ff_scale))
    print('Hidden Layer Size: {}'.format(hidden_size))
    print('Output Layer Depth: {}'.format(output_layer_depth))
    print('Number of Hidden Layers: {}'.format(num_hidden_layers))
    #print('Nesterov: {}'.format(nesterov))
    print('Adam')
    #print('Adadelta')

    net = GRUNeuralNetwork(21,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_layer_depth=output_layer_depth,
            hidden_scale=hidden_scale, ff_scale=ff_scale, 
            init_lr=init_lr, gamma=gamma, weight_decay=weight_decay)

    train_mses, val_mses = net.train(dataset, val_dataset, sys.argv[3], patience=patience, warm_start_params=warm_start_params,  warm_start_last_epoch=warm_start_last_epoch, max_iter=max_iter)

    os.makedirs(pred_dir, exist_ok=True)
    for i in range(len(dataset)):
        x, y, lengths, proteins = dataset[i]
        y_pred = net.predict(x, lengths)
        write_preds(proteins, y_pred, pred_dir)

    for i in range(len(val_dataset)):
        x, y, lengths, proteins = val_dataset[i]
        y_pred = net.predict(x, lengths)
        write_preds(proteins, y_pred, pred_dir)

    for i in range(len(test_dataset)):
        x, y, lengths, proteins = test_dataset[i]
        y_pred = net.predict(x, lengths)
        write_preds(proteins, y_pred, pred_dir)

    test_mean_mse = 0.0
    test_mean_pcc = 0.0
    for i in range(len(test_dataset)):
        x, y, lengths, _ = test_dataset[i]
        y_pred = net.predict(x, lengths)
        test_mean_pcc += get_avg_pcc(y, y_pred, lengths)
        test_mean_mse += get_avg_mse(y, y_pred, lengths)
    test_mean_pcc /= len(test_dataset)
    test_mean_mse /= len(test_dataset)
    print('Test MSE: {} PCC: {}'.format(test_mean_mse, test_mean_pcc))
    exit()
    #print(list(zip(train_mses, test_mses)))
    #train_nn(net, dataset, train_indices, validation_indices, sys.argv[3])
    #scores = cross_validation(net, dataset, indices, 10, 0.40)
    #print(scores)
    param_grid = {'init_lr' : 10.0 ** np.arange(-3,-1), 'hidden_size' : [16,32],
                    'weight_decay' : 10.0 ** np.arange(-6,-5), 'gamma' : [0.9],
                    'output_layer_depth' : [2,4],
                    'num_hidden_layers' : [2]}
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

    cv_fold = 10
    cv_mse_threshold = 0.85
    results = gridsearchcv(net, files, batch_size, cv_fold, cv_mse_threshold, param_grid, param_set_funcs)
    for i in range(len(results)):
        for param_name, param_value in results[i][0].items():
            print('{}:{}'.format(param_name, param_value), end=' ')
        print('score: {0:.4f}'.format(results[i][1]))
    #print(results)
    best_config = max(results, key=lambda t:t[1])
    print(best_config)
    
