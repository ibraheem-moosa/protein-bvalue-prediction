import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import time
import sys
import os
import random
import random
from scipy.stats import pearsonr
from Bio.PDB import Polypeptide

class FeedForward(nn.Module):
    def __init__(self, input_size, num_hidden_layers, width=None):
        super(FeedForward, self).__init__()
        self.num_layers = num_hidden_layers + 2
        self.fc = nn.ModuleList()
        if width is None:
            width = input_size
        self.fc.append(nn.Linear(input_size, width))
        for i in range(num_hidden_layers):
            self.fc.append(nn.Linear(width, width))
                    
        self.fc.append(nn.Linear(width, 1))
        self._init_weights_()

    def forward(self, x):
        for i in range(self.num_layers):
            x = relu(self.fc[i](x))
        return x
    
    def _init_weights_(self):
        gain = nn.init.calculate_gain('relu')
        init_method = nn.init.xavier_normal_
        bias_init_method = nn.init.constant_
        for i in range(self.num_layers):
            init_method(self.fc[i].weight.data, gain=gain)
            bias_init_method(self.fc[i].bias.data, 0)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


class SequentialFeedForward(nn.Module):
    def __init__(self, window_size, num_layers, nn_width):
        super(SequentialFeedForward, self).__init__()
        self.ws = window_size
        self.nn_input_size = self.ws * 21 + self.ws
        self.nn_num_layers = num_layers
        self.nn_width = nn_width
        self.base_nn = FeedForward(self.nn_input_size, self.nn_num_layers, width=nn_width)

    def forward(self, x):
        y_pred = [torch.zeros(1, dtype=torch.float) for i in range(self.ws)]
        x_window = [torch.zeros(21, dtype=torch.float) for i in range(self.ws)]
        for i in range(len(x)):
            x_window.pop(0)
            x_i = np.zeros(21, dtype=np.float32)
            x_i[x[i]] = 1.0
            x_i = torch.from_numpy(x_i)
            x_i.requires_grad = True
            x_window.append(x_i)
            y_pred.append(self.base_nn.forward(torch.cat(x_window + y_pred[i:i+self.ws])))
        return y_pred[self.ws:]

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def train(self, X, Y, init_lr, momentum, weight_decay, gamma, num_epochs):
        optimizer = torch.optim.SGD(self.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        criterion = nn.MSELoss()
        xy_pair = list(zip(X, Y))
        for epoch in range(num_epochs):
            running_loss = 0.0
            index = 0
            for x, y in xy_pair:
                optimizer.zero_grad()
                y_pred = self.forward(x)
                y_pred = torch.cat(y_pred)
                y_true = torch.from_numpy(y)
                loss = criterion(y_pred, y_true)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                index += 1
                if index % 250 == 0:
                    print("At sample {}".format(index))
            random.shuffle(xy_pair)
            print("Epoch: {} Running Loss: {} Average PCC: {}".format(epoch, running_loss, self.avg_pcc(X, Y)))
            print(time.strftime('%Y-%m-%d %H:%M:%S'))


    def mse(self, X, Y):
        criterion = nn.MSELoss()
        running_loss = 0.0
        for x, y in zip(X, Y):
            y_pred = self.predict(x)
            y_pred = torch.cat(y_pred)
            y_true = torch.from_numpy(y)
            loss = criterion(y_pred, y_true)
            running_loss += loss.item()
        return running_loss

    def avg_pcc(self, X, Y):
        pccs = []
        avg_val = 0.0
        for x, y in zip(X, Y):
            y_pred = self.predict(x)
            y_pred = torch.cat(y_pred)
            y_pred = y_pred.numpy()
            pcc = pearsonr(y_pred, y)[0]
            pccs.append(pcc)
            avg_val += np.mean(y_pred)
        print(avg_val / len(X))
        pccs = np.array(pccs)
        pccs = np.nan_to_num(pccs)
        return np.mean(pccs)



def summarize_tensor(tensor):
    return torch.min(tensor).item(), torch.max(tensor).item(), torch.mean(tensor).item(), torch.std(tensor).item()
def summarize_ndarray(array):
    return np.min(array), np.max(array), np.mean(array), np.std(array)
def summarize_ndarrays(arrays):
    mins, maxs, means, stds = tuple(zip(*list(map(summarize_ndarray, arrays))))
    return np.mean(mins), np.mean(maxs), np.mean(means), np.mean(stds)

def aa_to_index(aa):
    """
    :param aa: Three character amino acid name.
    :returns: Integer index as per BioPython, unknown/non-standard amino acids return 20.
    """
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python3 torch_nn.py data_dir protein_list')
        exit()
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    random.seed(42)
    protein_list = []
    with open(sys.argv[2]) as f:
        for line in f:
            line = line.strip()
            protein_list.append(line)
    data_dir = sys.argv[1]
    files = protein_list
    indices = list(range(len(files)))
    random.shuffle(indices)

    train_files = [files[i] for i in indices[:int(len(indices) * 0.80)]]
    val_files = [files[i] for i in indices[int(len(indices) * 0.80):]]
    train_X = []
    train_Y = []
    for fname in train_files:
        train_X.append([])
        train_Y.append([])
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                line = line.strip()
                aa, b = line.split()
                aa = aa_to_index(aa)
                b = float(b)
                train_X[-1].append(aa)
                train_Y[-1].append(b)
        train_Y[-1] = np.array(train_Y[-1], dtype=np.float32)

    val_X = []
    val_Y = []
    for fname in val_files:
        val_X.append([])
        val_Y.append([])
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                line = line.strip()
                aa, b = line.split()
                aa = aa_to_index(aa)
                b = float(b)
                val_X[-1].append(aa)
                val_Y[-1].append(b)
        val_Y[-1] = np.array(val_Y[-1], dtype=np.float32)

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(sum(map(len, train_Y)))
    ws = 5
    init_lr = 1e-2
    momentum = 0.9
    weight_decay = 100.0
    gamma = 0.99
    num_epochs = 10
    num_hidden_layers = 2
    nn_width = (ws * 21 + ws) // 10
    net = SequentialFeedForward(ws, num_hidden_layers, nn_width)
    net.train(train_X, train_Y, init_lr, momentum, weight_decay, gamma, num_epochs)
    val_mse = net.mse(val_X, val_Y)
    print("Validation Loss: {}".format(val_mse))

