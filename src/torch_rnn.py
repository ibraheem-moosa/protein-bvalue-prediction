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

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=8, output_layer_depth=1,  num_hidden_layers=1, hidden_scale=1.0, ff_scale=0.001, init_lr=1e-3, gamma=0.99, weight_decay=0.1, grad_clip=1.0):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_layer_depth = output_layer_depth
        self.num_hidden_layers = num_hidden_layers
        self.hidden_scale = hidden_scale
        self.ff_scale = ff_scale
        self.init_lr = init_lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.init_layers()

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        packed_out, h = self.rnn_layer(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.output_layer(out)
        return out

    def init_layers(self):
        self.rnn_layer = nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                nonlinearity='relu',
                                num_layers=self.num_hidden_layers,
                                batch_first=True, 
                                bidirectional=True)
        self.output_layer = FixedWidthFeedForwardNeuralNetwork(self.hidden_size * 2, 1, self.output_layer_depth, leaky_relu)
        self._init_weights_()

    def _init_weights_(self):
        ff_init_method = nn.init.normal_
        hidden_weight_init_method = nn.init.eye_
        bias_init_method = nn.init.constant_
        for name, param in self.rnn_layer.named_parameters():
            if 'weight_hh' in name:
                hidden_weight_init_method(param)
                with torch.no_grad():
                    param.mul_(self.hidden_scale)
                param.requires_grad_()
            elif 'weight_ih' in name:
                ff_init_method(param, std=self.ff_scale)
            else:
                bias_init_method(param, 0)

        for name, param in self.output_layer.named_parameters():
            if 'weight' in name:
                ff_init_method(param, std=self.ff_scale)
            else:
                bias_init_method(param, 0)

    def predict(self, x, lengths):
        with torch.no_grad():
            out = self.forward(x, lengths)
            return out

    def reset_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
        self.init_layers()

    def reset_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
        self.init_layers()

    def reset_gamma(self, gamma):
        self.gamma = gamma
        self.init_layers()

    def reset_output_layer_depth(self, output_layer_depth):
        self.output_layer_depth = output_layer_depth
        self.init_layers()
    
    def reset_num_hidden_layers(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers
        self.init_layers()

    def train(self, dataset, validation_dataset, model_dir=None, patience=5, warm_start_last_epoch=-1):
        #self.cuda()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam([{'params' : self.parameters(), 'initial_lr' : self.init_lr}], 
                        lr=self.init_lr, weight_decay=self.weight_decay, amsgrad=False)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.gamma)
        self._init_weights_()
        num_of_batches = len(dataset)
        validation_num_of_batches = len(validation_dataset)
        
        best_epoch = 0
        best_mse = 1000.0
        validation_mses = []
        validation_pccs = []
        train_mses = []
        train_pccs = []
        for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 1000):
            scheduler.step()
            for i in range(num_of_batches):
                x, y, lengths = dataset[i]
                optimizer.zero_grad()
                y_pred = self.forward(x, lengths)
                loss = criterion(y_pred, y)
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
                optimizer.step()

            train_loss = 0.0
            mean_pcc = 0.0
            for i in range(num_of_batches):
                x, y, lengths = dataset[i]
                y_pred = self.predict(x, lengths)
                loss = criterion(y_pred, y)
                train_loss += loss.item()
                mean_pcc += get_avg_pcc(y, y_pred, lengths)
            train_loss /= dataset.total_length()
            mean_pcc /= num_of_batches
            train_mses.append(train_loss)
            train_pccs.append(mean_pcc)

            validation_loss = 0.0
            mean_pcc = 0.0
            for i in range(validation_num_of_batches):
                x, y, lengths = validation_dataset[i]
                y_pred = self.predict(x, lengths)
                loss = criterion(y_pred, y)
                validation_loss += loss.item()
                mean_pcc += get_avg_pcc(y, y_pred, lengths)
            validation_loss /= validation_dataset.total_length()
            mean_pcc /= validation_num_of_batches
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
                torch.save(self.state_dict(), os.path.join(model_dir, 'net-{0:02d}'.format(epoch)))

            if epoch - best_epoch == patience:
                break

        return train_mses, validation_mses


class LSTMNeuralNetwork(RecurrentNeuralNetwork):
    def __init__(self, input_size, hidden_size=8, output_layer_depth=1,  num_hidden_layers=1, hidden_scale=1.0, ff_scale=0.001, init_lr=1e-3, gamma=0.99, weight_decay=0.1, grad_clip=1.0):
        super(LSTMNeuralNetwork, self).__init__(input_size, hidden_size, output_layer_depth, num_hidden_layers, hidden_scale, ff_scale, init_lr, gamma, weight_decay, grad_clip)

    def init_layers(self):
        self.lstm_layer = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                #nonlinearity='relu',
                                num_layers=self.num_hidden_layers,
                                batch_first=True, 
                                bidirectional=True)
        self.output_layer = FixedWidthFeedForwardNeuralNetwork(self.hidden_size * 2, 1, self.output_layer_depth, leaky_relu)
        self._init_weights_()

    def _init_weights_(self):
        ff_init_method = nn.init.normal_
        hidden_weight_init_method = nn.init.eye_
        bias_init_method = nn.init.constant_
        for name, param in self.lstm_layer.named_parameters():
            if 'weight_hh' in name:
                hidden_weight_init_method(param)
                with torch.no_grad():
                    param.mul_(self.hidden_scale)
                param.requires_grad_()
            elif 'weight_ih' in name:
                ff_init_method(param, std=self.ff_scale)
            else:
                bias_init_method(param, 0)

        for name, param in self.output_layer.named_parameters():
            if 'weight' in name:
                ff_init_method(param, std=self.ff_scale)
            else:
                bias_init_method(param, 0)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        packed_out, h = self.lstm_layer(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.output_layer(out)
        return out



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
    momentum = 0.9
    weight_decay = 1e-3
    gamma = 0.9
    hidden_size = 8
    hidden_scale = 0.1
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

    net = RecurrentNeuralNetwork(21,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_layer_depth=output_layer_depth,
            hidden_scale=hidden_scale, ff_scale=ff_scale, 
            init_lr=init_lr, gamma=gamma, 
            weight_decay=weight_decay, grad_clip=grad_clip)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    #train_nn(net, dataset, train_indices, validation_indices, sys.argv[3])
    #scores = cross_validation(net, dataset, indices, 10, 0.40)
    #print(scores)
    
    param_grid = {'init_lr' : 2.0 ** np.arange(-7,-6), 'hidden_size' : [8],
                    'weight_decay' : 10.0 ** np.arange(-3,-2), 'gamma' : [0.9],
                    'output_layer_depth' : [2],
                    'num_hidden_layers' : [1, 2, 3]}
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
    print(results)
