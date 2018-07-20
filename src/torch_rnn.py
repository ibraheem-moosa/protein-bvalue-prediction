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
    def __init__(self, input_size, hidden_size=8, output_layer_depth=1,  num_hidden_layers=1, hidden_scale=1.0, ff_scale=0.001):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_layer_depth = output_layer_depth
        self.num_hidden_layers = num_hidden_layers
        self.hidden_scale = hidden_scale
        self.ff_scale = ff_scale
        self.rnn_layer = nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                nonlinearity='relu',
                                num_layers=self.num_hidden_layers,
                                batch_first=True, 
                                bidirectional=True)
        self.output_layer = FixedWidthFeedForwardNeuralNetwork(hidden_size * 2, 1, output_layer_depth, leaky_relu)
        self._init_weights_(self.hidden_scale, self.ff_scale)

    def forward(self, x):
        out, h = self.rnn_layer(x)
        out = self.output_layer(out)
        return out

    def _init_weights_(self, hidden_scale=1.0, ff_scale=0.001):
        ff_init_method = nn.init.normal_
        hidden_weight_init_method = nn.init.eye_
        bias_init_method = nn.init.constant_
        for name, param in self.rnn_layer.named_parameters():
            if 'weight_hh' in name:
                hidden_weight_init_method(param)
                with torch.no_grad():
                    param.mul_(hidden_scale)
                param.requires_grad_()
            elif 'weight_ih' in name:
                ff_init_method(param, std=ff_scale)
            else:
                bias_init_method(param, 0)

        for name, param in self.output_layer.named_parameters():
            if 'weight' in name:
                ff_init_method(param, std=ff_scale)
            else:
                bias_init_method(param, 0)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            return out

    def reset_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                nonlinearity='relu',
                                num_layers=self.num_hidden_layers,
                                batch_first=True, 
                                bidirectional=True)
        self.output_layer = FixedWidthFeedForwardNeuralNetwork(hidden_size * 2, 1, output_layer_depth, leaky_relu)
        self._init_weights_(self.hidden_scale, self.ff_scale)


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

def train_nn(net, optimizer, criterion, scheduler, dataset, train_indices, validation_indices, model_dir=None, patience=10, warm_start_last_epoch=-1):
    best_validation_pcc_epoch = 0
    best_validation_pcc = 0.0
    validation_pccs = []
    train_pccs = []
    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 20):
        print(time.strftime('%Y-%m-%d %H:%M:%S'))

        scheduler.step()
        running_loss = 0.0
        random.shuffle(train_indices)
        for i in train_indices:
            x, y = dataset[i]
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        print('Epoch: {} done. Loss: {}'.format(
                                epoch, running_loss / len(dataset)))

        train_pcc = get_avg_pcc(net, dataset, train_indices)
        train_pccs.append(train_pcc)
        print('Train Avg PCC: {}'.format(train_pcc))

        validation_pcc = get_avg_pcc(net, dataset, validation_indices)
        validation_pccs.append(validation_pcc)
        print('Validation Avg PCC: {}'.format(validation_pcc))
        if validation_pcc > best_validation_pcc:
            best_validation_pcc = validation_pcc
            best_validation_pcc_epoch = epoch

        if model_dir is not None:
            torch.save(net.state_dict(), os.path.join(model_dir, 'net-{0:02d}'.format(epoch)))

        if epoch - best_validation_pcc_epoch == patience:
            break

    return net, train_pccs, validation_pccs

def cross_validation(net, optimizer, criterion, scheduler, dataset, indices, k, threshold):
    n = len(indices) // k
    r = len(indices) - n * k
    fold_lengths = [n + 1] * r + [n] * (k - r)
    cumulative_fl = [0]
    for fl in fold_lengths:
        cumulative_fl.append(cumulative_fl[-1] + fl)
    scores = []
    for i in range(k):
        net._init_weights_()
        print('Cross Validation Fold: {}'.format(i))
        train_indices = []
        validation_indices = []
        for j in range(k):
            if j == i:
                validation_indices.extend(indices[cumulative_fl[j]:cumulative_fl[j+1]])
            else:
                train_indices.extend(indices[cumulative_fl[j]:cumulative_fl[j+1]])
        net, train_pccs, validation_pccs = train_nn(net, optimizer, criterion, scheduler, dataset, train_indices, validation_indices) 
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

def gridsearchcv(net, optimizer, criterion, scheduler, dataset, indices, k, threshold, param_grid, param_set_funcs):
    result = []
    num_of_params = len(param_grid)
    for param_config in get_param_config(param_grid, list(param_grid.keys())):
        next_param_config = param_config
        param_config_dict = dict()
        while True:
            key, value, next_param_config = next_param_config
            param_config_dict[key] = value
            param_set_funcs[key](net, optimizer, scheduler, value)
            if next_param_config is None:
                break
            
        print('Running CV for params {}'.format(param_config_dict))
        scores = cross_validation(net, optimizer, criterion, scheduler, dataset, indices, k, threshold)
        mean_score = sum(scores) / len(scores)
        result.append((param_config_dict, mean_score))
    return result

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
    indices = indices[:1000]
    train_indices = indices[:800]
    validation_indices = indices[800:]

    if len(sys.argv) == 6:
        warm_start_model_params = torch.load(sys.argv[4])
        warm_start_last_epoch = int(sys.argv[5])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 1
    init_lr = 0.008
    momentum = 0.9
    weight_decay = 1e-7
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

    net = RecurrentNeuralNetwork(21, hidden_size=hidden_size, hidden_scale=hidden_scale, 
            num_hidden_layers=num_hidden_layers, ff_scale=ff_scale, output_layer_depth=output_layer_depth)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    criterion = nn.MSELoss()
    #optimizer = optim.SGD([{'params' : net.parameters(), 'initial_lr' : init_lr}],
    #                        lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}], 
                            lr=init_lr, weight_decay=weight_decay, amsgrad=False)
    #optimizer = optim.Adadelta([{'params' : net.parameters(), 'initial_lr' : init_lr}], 
    #                        lr=init_lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    #train_nn(net, optimizer, criterion, scheduler, dataset, train_indices, validation_indices, sys.argv[3])
    #scores = cross_validation(net, optimizer, criterion, scheduler, dataset, indices, 10, 0.40)
    #print(scores)
    
    param_grid = {'init_lr' : 10.0 ** np.arange(-5,0), 'hidden_size' : 2 ** np.arange(0, 7)}
    def set_init_lr(net, optimizer, scheduler, value):
        optimizer.init_lr = value
    def set_hidden_size(net, optimizer, scheduler, value):
        net.reset_hidden_size(value)
    param_set_funcs = {'init_lr' : set_init_lr, 'hidden_size' : set_hidden_size}
    results = gridsearchcv(net, optimizer, criterion, scheduler, dataset, indices, 5, 0.40, param_grid, param_set_funcs)
    print(results)
