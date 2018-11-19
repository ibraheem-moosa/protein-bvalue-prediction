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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import sys
import time
import os
import random
from torch_helper import *
from torch_rnn_dataset import *


class CNN(nn.Module):
    def __init__(self, num_layers, input_size, kernel_size, chanel_size, init_lr, gamma, weight_decay, momentum=0):
        super(CNN, self).__init__()
        self.activation = torch.sigmoid
        #self.activation = relu
        self.init_lr = init_lr
        self.momentum = momentum
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.input_size = input_size
        self.conv_layers = []
        self.kernel_size = kernel_size
        self.chanel_size = chanel_size
        self._init_layers_()

    def _init_layers_(self):
        kernel_sizes = [self.kernel_size] * self.num_layers
        chanel_sizes = [self.input_size] + [self.chanel_size] * (self.num_layers - 1) + [1]
        del self.conv_layers
        self.conv_layers = []
        for i in range(self.num_layers):
            self.conv_layers.append(nn.Conv1d(chanel_sizes[i], chanel_sizes[i+1], kernel_sizes[i], padding=kernel_sizes[i] // 2))
        for i in range(self.num_layers):
            self.register_parameter('Conv1d-{0:3d}-_weight_'.format(i), self.conv_layers[i].weight)
            self.register_parameter('Conv1d-{0:3d}-_bias_'.format(i), self.conv_layers[i].bias)

    def forward(self, x):
        out = x
        for i in range(self.num_layers - 1):
            out = self.activation(self.conv_layers[i](out))
        out = self.conv_layers[-1](out)
        out = out.reshape((1, -1, 1))
        return out

    def predict(self, x):
        x = x.reshape((1,self.input_size,-1))
        with torch.no_grad():
            out = self.forward(x)
            return out

    def _init_weights_(self):
        for name, param in self.named_parameters():
            #print('Initializing parameter {}'.format(name))
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
                #nn.init.normal_(param, std=0.1)
            else:
                #nn.init.constant_(param, 0)
                nn.init.normal_(param, std=0.1)

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

    def shift(self, y_pred, y_true):
        y_true_lshift = torch.zeros(y_true.shape)
        y_true_rshift = torch.zeros(y_true.shape)
        y_true_lshift[:-1] = y_true[1:]
        y_true_rshift[1:] = y_true[:-1]
        y_pred_lshift = torch.zeros(y_pred.shape)
        y_pred_rshift = torch.zeros(y_pred.shape)
        y_pred_lshift[:-1] = y_pred[1:]
        y_pred_rshift[1:] = y_pred[:-1]
        lshift_del = y_pred_lshift - y_true_lshift
        rshift_del = y_pred_rshift - y_true_rshift
        return torch.sum(lshift_del * lshift_del) + torch.sum(rshift_del * rshift_del)
 
    def pcc(self, y_pred, y_true):
        y_pred -= y_pred.mean()
        y_true -= y_true.mean()
        return - torch.sum(y_pred * y_true) * torch.rsqrt(torch.sum(y_pred * y_pred) * torch.sum(y_true * y_true))

    def train(self, dataset, train_indices, validation_indices, model_dir=None, num_epochs=1000, patience=None, warm_start_last_epoch=-1):
        if patience == None:
            patience = num_epochs
        #self.cuda()
        criterion = nn.MSELoss()
        criterion = self.pcc
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), 
                        lr=self.init_lr, weight_decay=self.weight_decay, amsgrad=False)
        #optimizer = optim.SGD(self.parameters(),
        #        lr=self.init_lr, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=True)
        #optimizer = optim.ASGD(self.parameters(),
        #        lr=self.init_lr, weight_decay=self.weight_decay)
        #optimizer = optim.Adagrad(self.parameters(),
        #        lr=self.init_lr, weight_decay=self.weight_decay, lr_decay=0.1)
 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.gamma)
        self._init_weights_()
        
        best_epoch = 0
        best_pcc = -1000.0
        validation_mses = []
        validation_pccs = []
        train_mses = []
        train_pccs = []
        for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + num_epochs):
            scheduler.step()
            random.shuffle(train_indices)
            for i in train_indices:
                x, y = dataset[i]
                x = x.reshape((1,self.input_size,-1))
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = criterion(y_pred, y)# + criterion2(y_pred, y)
                loss.backward()
                #nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
                optimizer.step()

            mean_pcc = 0.0
            mean_mse = 0.0
            for i in train_indices:
                x, y = dataset[i]
                y_pred = self.predict(x)
                mean_pcc += pearsonr(y_pred.numpy().flatten(), y.numpy().flatten())[0]
                mean_mse += mean_squared_error(y_pred.numpy().flatten(), y.numpy().flatten())
            mean_pcc /= len(train_indices)
            mean_mse /= len(train_indices)
            train_mses.append(mean_mse)
            train_pccs.append(mean_pcc)
            if mean_pcc > best_pcc:
                best_pcc = mean_pcc
                best_epoch = epoch
                print(best_epoch)
 
            mean_pcc = 0.0
            mean_mse = 0.0
            for i in validation_indices:
                x, y = dataset[i]
                y_pred = self.predict(x)
                #if random.random() < 0.001:
                #    plot_true_and_prediction(y.numpy().flatten(), y_pred.numpy().flatten())
                mean_pcc += pearsonr(y_pred.numpy().flatten(), y.numpy().flatten())[0]
                mean_mse += mean_squared_error(y_pred.numpy().flatten(), y.numpy().flatten())
            mean_pcc /= len(validation_indices)
            mean_mse /= len(validation_indices)
            validation_mses.append(mean_mse)
            validation_pccs.append(mean_pcc)

           
            print('Epoch: {0:02d} Time: {1} MSE: {2:.6f} Test MSE: {3:.6f} PCC: {4:0.6f} Test PCC: {5:0.6f}'.format(
                                    epoch, time.strftime('%Y-%m-%d %H:%M:%S'), 
                                    train_mses[-1], validation_mses[-1], 
                                    train_pccs[-1], validation_pccs[-1]))

            for name, param in self.named_parameters():
                print(name)
                print(summarize_tensor(param))
                print(summarize_tensor(param.grad))
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

    init_lr = 1e-4
    momentum = 0.9
    weight_decay = 0.0
    gamma = 0.9
    num_layers = 15
    kernel_size = 3
    chanel_size = 3
    print('Initial LR: {}'.format(init_lr))
    print('Weight Decay: {}'.format(weight_decay))
    print('Number of Layers: {}'.format(num_layers))
    print('Kernel Size: {}'.format(kernel_size))
    print('Chanel Size: {}'.format(chanel_size))
    net = CNN(num_layers, 21, kernel_size, chanel_size, init_lr, gamma, weight_decay, momentum=momentum)
    net.train(dataset, train_indices, validation_indices, num_epochs=1000, patience=20)
    exit()
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    param_grid = {'init_lr' : 10.0 ** np.arange(-4, -3), 'num_layers' : [4],
                    'weight_decay' : 10.0 ** np.arange(-1, 0), 'gamma' : [0.9],
                    'chanel_size' : [3],
                    'kernel_size' : [7]}
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
        net.reset_kernel_size(kernel_size)

    param_set_funcs = {'init_lr' : set_init_lr, 'num_layers' : set_num_layers, 
                        'weight_decay' : set_weight_decay, 'gamma' : set_gamma, 
                        'chanel_size' : set_chanel_size,
                        'kernel_size' : set_kernel_size}

    results = gridsearchcv(net, dataset, indices, None, None, param_grid, param_set_funcs)
    results.sort(key=lambda t:t[1])
    print(results)
    best_param_config_dict = results[0][0]
    for name, value in best_param_config_dict.items():
        param_set_funcs[name](net, value)
    net.train(dataset, train_indices, validation_indices, num_epochs=1000)
    
