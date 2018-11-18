import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
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
from torch_rnn_dataset import *


class CNN(nn.Module):
    def __init__(self, num_layers, input_size, kernel_size, chanel_size, init_lr, gamma, weight_decay):
        self.activation = relu
        self.init_lr = init_lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.conv_layers = []
        self.kernel_size = kernel_size
        self.chanel_size = chanel_size
        self._init_layers_()

    def _init_layers_(self):
        self.kernel_sizes = [kernel_size] * 8
        self.chanel_sizes = [input_size] + [chanel_size] * 8
        for i in range(num_layers):
            self.conv_layers.append(nn.Conv1d(chanel_sizes[i], chanel_sizes[i+1], kernel_sizes[i], padding=kernel_sizes[i] // 2))


    def forward(self, x):
        out = self.activation(self.conv_layers[0](x))
        for i in range(1, num_layers - 1):
            out = self.activation(self.conv_layers[i](x))
        out = self.conv_layers[-1](x)
        return out

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            return out


    def _init_weights_(self):
        for name, param in self.conv_layer.named_parameters():
            print('Initializing parameter {}'.format(name))
            nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))

    def reset_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def reset_gamma(self, gamma):
        self.gamma = gamma

    def reset_num_layers(self, num_layers):
        self.num_layers = num_layers
        self._init_layers_()

    def reset_chanel_size(self, chanel_size):
        self.chanel_size = chanel_size
        self._init_layers_()

    def reset_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
        self._init_layers_()

    def train(self, dataset, train_indices, validation_indices, model_dir=None, num_epochs=1000, patience=None, warm_start_last_epoch=-1):
        if patience == None:
            patience = num_epochs
        #self.cuda()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam( 
                        lr=self.init_lr, weight_decay=self.weight_decay, amsgrad=False)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.gamma)
        self._init_weights_()
        validation_num_of_batches = len(validation_dataset)
        
        best_epoch = 0
        best_mse = 1000.0
        validation_mses = []
        validation_pccs = []
        train_mses = []
        train_pccs = []
        for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + num_epochs):
            scheduler.step()
            random.shuffle(train_indices)
            for i in train_indices:
                x, y = dataset[i]
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = criterion(y_pred, y)
                loss.backward()
                #nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
                optimizer.step()

            train_loss = 0.0
            mean_pcc = 0.0
            for i in train_indices:
                x, y = dataset[i]
                y_pred = self.predict(x)
                loss = criterion(y_pred, y)
                train_loss += loss.item()
                mean_pcc += pearsonr(y, y_pred)[0]
            train_loss /= len(train_indices)
            mean_pcc /= len(train_indices)
            train_mses.append(train_loss)
            train_pccs.append(mean_pcc)

            validation_loss = 0.0
            mean_pcc = 0.0
            for i in validation_indices:
                x, y = dataset[i]
                y_pred = self.predict(x)
                loss = criterion(y_pred, y)
                validation_loss += loss.item()
                mean_pcc += pearsonr(y, y_pred)[0]
            validation_loss /= len(validation_indices)
            mean_pcc /= len(validation_indices)
            validation_mses.append(validation_loss)
            validation_pccs.append(mean_pcc)

            if train_loss < best_mse:
                best_mse = train_loss
                best_epoch = epoch
                print(best_epoch)
            
            print('Epoch: {0:02d} Time: {1} Loss: {2:.6f} Test Loss: {3:.6f} PCC: {4:0.6f} Test PCC: {5:0.6f}'.format(
                                    epoch, time.strftime('%Y-%m-%d %H:%M:%S'), 
                                    train_loss, validation_loss, 
                                    train_pccs[-1], validation_pccs[-1]))

            if model_dir is not None:
                state_to_save = {'state_dict': self.state_dict(), 'optim': optimizer.state_dict()}
                torch.save(self.state_dict(), os.path.join(model_dir, 'net-{0:03d}'.format(epoch)))

            if epoch - best_epoch == patience:
                break

        return train_mses, validation_mses


class FixedWidthFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, width, num_outputs, num_layers, activation):
        super(FixedWidthFeedForwardNeuralNetwork, self).__init__()
        self.linear_layers = [nn.Linear(width, width) for i in range(num_layers-1)]
        self.linear_layers.append(nn.Linear(width, num_outputs))
        self.activation = activation
        for i in range(num_layers):
            self.register_parameter('FF' + str(i) + '_weight_', self.linear_layers[i].weight)
            self.register_parameter('FF' + str(i) + '_bias_', self.linear_layers[i].bias)

    def forward(self, x):
        out = self.activation(self.linear_layers[0](x))
        for i in range(1, len(self.linear_layers) - 1):
            out = self.activation(self.linear_layers[i](out))
        out = self.linear_layers[-1](out)
        return out


if __name__ == '__main__':

    torch.set_printoptions(precision=2, linewidth=140)
    torch.manual_seed(42)
    random.seed(42)

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
    random.shuffle(indices)
    #indices = indices[:500]
    train_indices = indices[:int(0.8 * len(indices))]
    validation_indices = indices[int(0.8 * len(indices)):]

    if len(sys.argv) == 6:
        warm_start_model_params = torch.load(sys.argv[4])
        warm_start_last_epoch = int(sys.argv[5])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 1

    init_lr = 1.0 / 128
    weight_decay = 1e-3
    gamma = 0.9
    num_layers = 8
    kernel_size = 3
    chanel_size = 3

    print('Initial LR: {}'.format(init_lr))
    print('Weight Decay: {}'.format(weight_decay))
    print('Number of Layers: {}'.format(num_layers))
    print('Adam')

    net = CNN(num_layers, kernel_sizes, chanel_sizes, init_lr, gamma, weight_decay)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    param_grid = {'init_lr' : 10.0 ** np.arange(-5, 0), 'num_layers' : [8],
                    'weight_decay' : 10.0 ** np.arange(-3,-2), 'gamma' : [0.9],
                    'chanel_size' : [3],
                    'kernel_size' : [3]}
    def set_init_lr(net, value):
        net.init_lr = value
    def set_weight_decay(net, weight_decay):
        net.reset_weight_decay(weight_decay)
    def set_gamma(net, gamma):
        net.reset_gamma(gamma)
    def set_num_layers(net, num_layers):
        net.reset_num_layers(num_layers)
    def set_chanel_size(net, chanel_size):
        net.reset_chanel_size(chanel_size)
    def set_kernel_size(net, kernel_size):
        net.reset_kernel_size(net, kernel_size)

    param_set_funcs = {'init_lr' : set_init_lr, 'num_layers' : set_num_layers, 
                        'weight_decay' : set_weight_decay, 'gamma' : set_gamma, 
                        'chanel_size' : set_chanel_size,
                        'kernel_size' : set_kernel_size}

    results = gridsearchcv(net, dataset, indices, None, None, param_grid, param_set_funcs)
    results.sort(key=lambda t:t[1])
    print(results)
    best_param_config_dict = results[0][0]
    for name, value in best_param_config_dict:
        param_set_funcs[name](net, value)
    net.train(dataset, train_indices, validation_indices, num_epochs=1000)
    
