import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=8, hidden_scale=1.0, ff_scale=0.001):
        super(RecurrentNeuralNetwork, self).__init__()
        self.layer1 = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu', batch_first=True, bidirectional=False)
        self.layer2 = nn.Linear(hidden_size, 1)
        self._init_weights_()

    def forward(self, x):
        out, h = self.layer1(x, torch.zeros(1, 1, self.layer1.hidden_size))
        out = self.layer2(out)
        return out
    
    def _init_weights_(self):
        ff_init_method = nn.init.normal_
        init_method = nn.init.constant_
        bias_init_method = nn.init.constant_
        torch.eye(hidden_size, out=self.layer1.weight_hh_l0, requires_grad=True)
        with  torch.no_grad():
            self.layer1.weight_hh_l0.mul_(hidden_scale)
        bias_init_method(self.layer1.bias_hh_l0, 0)
        ff_init_method(self.layer1.weight_ih_l0, std=ff_scale)
        bias_init_method(self.layer1.bias_ih_l0, 0)
        ff_init_method(self.layer2.weight, std=ff_scale)
        bias_init_method(self.layer2.bias, 0)

    def predict(self, x):
        with torch.no_grad():
            out, h = self.layer1(x, torch.zeros(1, 1, self.layer1.hidden_size))
            out = self.layer2(out)
            return out

class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files):
        self._protein_lengths = []
        self._Xes = []
        self._yes = []
        for xf,yf in zip(X_files, y_files):
            X = torch.from_numpy(scsp.load_npz(xf).toarray()).reshape((-1, 21))
            y = torch.from_numpy(np.load(yf)['y']).reshape((-1, 1))
            assert(X.shape[0] == len(y))
            self._protein_lengths.append(len(y))
            self._Xes.append(X)
            self._yes.append(y)

    def __getitem__(self, idx):
        return self._Xes[idx], self._yes[idx]

    def __len__(self):
        return len(self._protein_lengths)

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
    indices = list(range(len(X_files)))
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4)

    init_lr = 0.001
    momentum = 0.9
    weight_decay = 0.0
    hidden_size = 8
    hidden_scale = 1.0
    ff_scale = 0.1
    grad_clip = 1000.0
    nesterov = True

    print('Initial LR: {}'.format(init_lr))
    #print('Momentum: {}'.format(momentum))
    print('Weight Decay: {}'.format(weight_decay))
    print('Gradient Clip: {}'.format(grad_clip))
    print('Hidden Scale: {}'.format(hidden_scale))
    print('FF Scale: {}'.format(ff_scale))
    print('Hidden Layer Size: {}'.format(hidden_size))
    #print('Nesterov: {}'.format(nesterov))
    print('Adam')

    net = RecurrentNeuralNetwork(21, hidden_size, hidden_scale, ff_scale)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    from math import sqrt
    import torch.optim as optim
    criterion = nn.MSELoss()
    optimizer = optim.SGD([{'params' : net.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}], 
                            lr=init_lr, weight_decay=weight_decay)

    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 500):
        running_loss = 0.0
        layer1_hidden_grad = torch.zeros_like(net.layer1.weight_hh_l0)
        layer1_hidden_bias_grad = torch.zeros_like(net.layer1.bias_hh_l0)
        layer1_input_grad = torch.zeros_like(net.layer1.weight_ih_l0)
        layer1_input_bias_grad = torch.zeros_like(net.layer1.bias_ih_l0)
        layer2_bias_grad = torch.zeros_like(net.layer2.bias)
        layer2_grad = torch.zeros_like(net.layer2.weight)
        for i, data in enumerate(dataloader):
            x, y = data
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)
            layer1_hidden_bias_grad += net.layer1.bias_hh_l0.grad
            layer1_hidden_grad += net.layer1.weight_hh_l0.grad
            layer1_input_bias_grad += net.layer1.bias_ih_l0.grad
            layer1_input_grad += net.layer1.weight_ih_l0.grad
            layer2_bias_grad += net.layer2.bias.grad
            layer2_grad += net.layer2.weight.grad
            optimizer.step()
            running_loss += loss.item()

        layer1_hidden_grad /= len(dataset)
        layer1_hidden_bias_grad /= len(dataset)
        layer1_input_grad /= len(dataset)
        layer1_input_bias_grad /= len(dataset)
        layer2_grad /= len(dataset)
        layer2_bias_grad /= len(dataset)

        print(time.strftime('%Y-%m-%d %H:%M'))
        print('Epoch: {} done. Loss: {}'.format(
                                epoch, running_loss / len(dataset)))
        print('\tInput Weight Grad: {}'.format(layer1_input_grad))
        print('\tInput Weight: {}'.format(net.layer1.weight_ih_l0))
        print('\tInput Bias Grad: {}'.format(layer1_input_bias_grad))
        print('\tInput Bias: {}'.format(net.layer1.bias_ih_l0))
        
        print('\tHidden Weight Grad: {}'.format(layer1_hidden_grad))
        print('\tHidden Weight: {}'.format(net.layer1.weight_hh_l0))
        print('\tHidden Bias Grad: {}'.format(layer1_hidden_bias_grad))
        print('\tHidden Bias: {}'.format(net.layer1.bias_hh_l0))
        
        print('\tOutput Weight Grad: {}'.format(layer2_grad))
        print('\tOutput Weight: {}'.format(net.layer2.weight))
        print('\tOutput Bias Grad: {}'.format(layer2_bias_grad))
        print('\tOutput Bias: {}'.format(net.layer2.bias))

        validation_pcc = []
        for x, y in validation_dataloader:
            y_pred = net.predict(x)
            validation_pcc.append(pearsonr(y_pred.numpy().flatten(), y.numpy().flatten())[0])
        validation_pcc = np.array(validation_pcc)
        validation_pcc[np.isnan(validation_pcc)] = 0
        train_pcc = []
        for x, y in dataloader:
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

