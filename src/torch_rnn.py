import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect

class FixedWidthFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, width, num_outputs, num_layers, activation):
        super(FixedWidthFeedForwardNeuralNetwork, self).__init__()
        self.linear_layers = [nn.Linear(width, width) for i in range(num_layers-1)]
        self.linear_layers.append(nn.Linear(width, num_outputs))
        self.activation = activation
        
    def forward(self, x):
        out = self.activation(self.linear_layers[0](x))
        for i in range(1, len(self.linear_layers) - 1):
            out = self.activation(self.linear_layers[i](out))
        out = self.linear_layers[-1](out)
        return out

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=8, hidden_scale=1.0, num_hidden_layers=1, ff_scale=0.001, output_layer_depth=1):
        super(RecurrentNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, 
                nonlinearity='relu', num_layers=num_hidden_layers, batch_first=True, bidirectional=False)
        self.output_layer = FixedWidthFeedForwardNeuralNetwork(hidden_size, 1, output_layer_depth, leaky_relu)
        self._init_weights_()

    def forward(self, x):
        out, h = self.rnn_layer(x, torch.zeros(self.num_hidden_layers, x.shape[0], self.hidden_size))
        out = self.output_layer(out)
        return out
    
    def _init_weights_(self):
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


class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files):
        self._Xes = []
        self._yes = []
        for xf,yf in zip(X_files, y_files):
            X = torch.from_numpy(scsp.load_npz(xf).toarray()).reshape((-1, 21))
            y = torch.from_numpy(np.load(yf)['y']).reshape((-1, 1))
            assert(X.shape[0] == y.shape[0])
            self._Xes.append(X)
            self._yes.append(y)
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
        
    def __getitem__(self, idx):
        return self._Xes[idx], self._yes[idx]

    def __len__(self):
        return len(self._Xes)

def summarize_tensor(tensor):
    return torch.max(tensor).item(), torch.min(tensor).item(), torch.mean(tensor).item(), torch.std(tensor).item()

if __name__ == '__main__':

    torch.set_printoptions(precision=2, linewidth=140)
    torch.manual_seed(42)
    import sys

    if len(sys.argv) < 7:
        print('Usage: python3 regression.py data_dir X_files y_files validation_dir X_validation_files y_validation_files checkpoint_dir [warm_start_model] [warm_start_epoch]')
        exit()

    import time
    print(time.strftime('%Y-%m-%d %H:%M'))

    import os
    X_files_list = open(sys.argv[2])
    X_files = []
    for line in X_files_list:
        X_files.append(os.path.join(sys.argv[1], line[:-1]))
    y_files_list = open(sys.argv[3])
    y_files = []
    for line in y_files_list:
        y_files.append(os.path.join(sys.argv[1], line[:-1]))

    X_validation_files_list = open(sys.argv[5])
    X_validation_files = []
    for line in X_validation_files_list:
        X_validation_files.append(os.path.join(sys.argv[4], line[:-1]))
    y_validation_files_list = open(sys.argv[6])
    y_validation_files = []
    for line in y_validation_files_list:
        y_validation_files.append(os.path.join(sys.argv[4], line[:-1]))


    import random
    indices = list(range(len(X_files)))
    random.shuffle(indices)

    X_files = [X_files[i] for i in indices]
    y_files = [y_files[i] for i in indices]
    X_validation_files = X_files[-len(indices) // 5:]
    y_validation_files = y_files[-len(indices) // 5:]
    X_files = X_files[:-len(indices) // 5]
    y_files = y_files[:-len(indices) // 5]
    dataset = ProteinDataset(X_files, y_files)
    print('Dataset init done ', len(dataset))
    validation_dataset = ProteinDataset(X_validation_files, y_validation_files)
    print('Validation dataset init done ', len(validation_dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))

    if len(sys.argv) == 10:
        warm_start_model_params = torch.load(sys.argv[8])
        warm_start_last_epoch = int(sys.argv[9])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 1
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    #validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4)

    init_lr = 0.005
    momentum = 0.9
    weight_decay = 0.001
    hidden_size = 8
    hidden_scale = 1.0
    num_hidden_layers = 4
    output_layer_depth = 4
    ff_scale = 0.1
    grad_clip = 1.0
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
    print('Three hidden layer at output')
    #print('Nesterov: {}'.format(nesterov))
    print('Adam, Amsgrad')
    #print('Adadelta')

    net = RecurrentNeuralNetwork(21, hidden_size, hidden_scale, num_hidden_layers, ff_scale, output_layer_depth)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    from math import sqrt
    import torch.optim as optim
    criterion = nn.MSELoss()
    #optimizer = optim.SGD([{'params' : net.parameters(), 'initial_lr' : init_lr}],
    #                        lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}], 
                            lr=init_lr, weight_decay=weight_decay, amsgrad=True)
    #optimizer = optim.Adadelta([{'params' : net.parameters(), 'initial_lr' : init_lr}], 
    #                        lr=init_lr, weight_decay=weight_decay)


    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    indices = list(range(len(dataset)))
    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 500):
        #scheduler.step()
        running_loss = 0.0
        random.shuffle(indices)
        for i in indices:
            x, y = dataset[i]
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        print(time.strftime('%Y-%m-%d %H:%M'))
        print('Epoch: {} done. Loss: {}'.format(
                                epoch, running_loss / len(dataset)))
        print('##############################################################')
        for name, param in net.named_parameters():
            print('---------------------------------------------------------------')
            print(name)
            print(summarize_tensor(param))
        print('##############################################################')

        '''
        print('\tInput Weight Grad: {}'.format(summarize_tensor(layer1_input_grad)))
        print('\tInput Weight: {}'.format(summarize_tensor(net.layer1.weight_ih_l0)))
        print('\tInput Bias Grad: {}'.format(summarize_tensor(layer1_input_bias_grad)))
        print('\tInput Bias: {}'.format(summarize_tensor(net.layer1.bias_ih_l0)))
        
        print('\tHidden Weight Grad: {}'.format(summarize_tensor(layer1_hidden_grad)))
        print('\tHidden Weight: {}'.format(summarize_tensor(net.layer1.weight_hh_l0)))
        print('\tHidden Bias Grad: {}'.format(summarize_tensor(layer1_hidden_bias_grad)))
        print('\tHidden Bias: {}'.format(summarize_tensor(net.layer1.bias_hh_l0)))
        
        print('\tOutput Weight Grad: {}'.format(summarize_tensor(layer2_grad)))
        print('\tOutput Weight: {}'.format(summarize_tensor(net.layer2.weight)))
        print('\tOutput Bias Grad: {}'.format(summarize_tensor(layer2_bias_grad)))
        print('\tOutput Bias: {}'.format(summarize_tensor(net.layer2.bias)))
        '''

        validation_pcc = []
        for i in range(len(validation_dataset)):
            x, y = validation_dataset[i]
            y_pred = net.predict(x)
            validation_pcc.append(pearsonr(y_pred.numpy().flatten(), y.numpy().flatten())[0])
        validation_pcc = np.array(validation_pcc)
        validation_pcc[np.isnan(validation_pcc)] = 0
        train_pcc = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            y_pred = net.predict(x)
            train_pcc.append(pearsonr(y_pred.numpy().flatten(), y.numpy().flatten())[0])
        train_pcc = np.array(train_pcc)
        train_pcc[np.isnan(train_pcc)] = 0

        '''
        R2 = 1 - (mean squared error) / (Variance of truth)
        '''
        print('Validation Avg PCC: {}'.format(np.mean(validation_pcc)))
        print('Train Avg PCC: {}'.format(np.mean(train_pcc)))
        print(time.strftime('%Y-%m-%d %H:%M'))
    #    for g in optimizer.param_groups:
    #        g['lr'] = init_lr / sqrt(epoch + 1)
        if (epoch + 1) % 20 == 0:
            torch.save(net.state_dict(), os.path.join(sys.argv[7], 'net-{0:02d}'.format(epoch)))

    save_net = input('Do you want to save the net?')
    if 'y' in save_net:
        fname = input('File Name;')
        torch.save(net.state_dict(), fname)

