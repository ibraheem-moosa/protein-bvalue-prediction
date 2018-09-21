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
from Bio.PDB import Polypeptide

class FeedForward(nn.Module):
    def __init__(self, input_size, num_layers):
        super(FeedForward, self).__init__()
        self.num_layers = num_layers
        self.fc = list()
        for i in range(num_layers - 1):
            self.fc.append(
                    nn.Linear(input_size, input_size)
        self.fc.append(nn.Linear(input_size, 1))
        self._init_weights_()

    def forward(self, x):
        for i in range(self.num_layers):
            x = leaky_relu(self.fc[i](x))
        return x
    
    def _init_weights_(self):
        gain = nn.init.calculate_gain('leaky_relu')
        init_method = nn.init.xavier_normal_
        bias_init_method = nn.init.constant_
        for i in range(self.num_layers):
            init_method(self.fc[i].weight.data, gain=gain)
            bias_init_method(self.fc[i].bias.data, 0)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def calculate_mse(self, dataset):
        criterion = nn.MSELoss(reduction='sum')
        total_mse = 0.0
        while dataset.has_next():
            X, y = dataset.next()
            y_pred = self.predict(X)
            loss = criterion(y_pred, y)
            total_mse += loss.item()
        dataset.reset()
        return total_mse / dataset.length()

    def train(self, num_iter, epoch_per_iter, dataset, validation_dataset):
        init_lr = 0.0005
        batch_size = 32
        optimizer = torch.optim.Adam([{'params' : self.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        criterion = nn.MSELoss()
        for i in range(num_iter):
            for epoch in range(epoch_per_iter):
                scheduler.step()
                while dataset.has_next():
                    X, y = dataset.next(batch_size)
                    optimizer.zero_grad()
                    y_pred = self.forward(X)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()

                dataset.reset()
                train_mse = self.calculate_mse(dataset)
                val_mse = self.calculate_mse(validation_dataset)
                print("Iter: {} Epoch: {} TL: {} VL: {}".format(num_iter, epoch, train_mse, val_mse))

            while dataset.has_next():
                X, y = dataset.next(batch_size)
                y_pred = self.predict(X)
                dataset.set_y_pred(y_pred)
            dataset.reset()


class Dataset:

    def __init__(self, X, y, ws):
        self.lengths = list(map(len, y))
        self.length = sum(lengths)
        self.ws = ws
        self.y = []
        index = [[], []]
        value = []
        num_column = 21 * (2 * ws + 1) + ws
        current_row = 0
        for i in range(len(X)):
            self.y.extend(y[i])
            aa_in_window = [20] * ws + X[i][:ws + 1]
            for j in range(len(X[i])):
                value.extend([1] * (2 * ws + 1))
                value.extend([0] * ws)
                index[0].extend([current_row] * (3 * ws + 1))
                index[1].extend(list(map(lambda t: 21 * t[0] + t[1], enumerate(aa_in_window))))
                index[1].extend(list(range(num_column - ws, num_column)))
                current_row += 1
                aa_in_window.pop(0)
                aa_in_window.append(21 if j + ws > len(X[i]) else X[i][j + ws])
                assert(len(aa_in_window) == 2 * ws + 1)
        self.X = torch.sparse.FloatTensor(index, value, torch.Size([current_row, num_column]))
        self.y = torch.FloatTensor(self.y)
        self.cursor = 0
        self.num_row = current_row
        self.num_col = num_column
        self.nonzero_col = 3 * ws + 1

    def length(self):
        return self.length

    def has_next(self):
        if self.cursor == self.num_row:
            return False
        return True

    def reset(self):
        self.cursor = 0

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = min(self.num_row, 
                    (1024 * 1024 * 1024) // (self.num_col * 8))
        if batch_size == self.num_row:
            self.cursor = self.num_row
            return self.X.to_dense(), self.y
        if self.cursor + batch_size >= self.num_row:
            batch_size = self.num_row - self.cursor

        start = self.cursor * self.nonzero_col
        end = start + batch_size * self.nonzero_col
        index = self.X._indices()
        index = [index[0][start:end], index[1][start:end]]
        value = self._values()
        value = values[start:end]
        self.cursor += batch_size
        return torch.sparse.FloatTensor(index, value, 
                torch.Size([batch_size, self.num_col])).to_dense(), 
                y[self.cursor, self.cursor + batch_size]

    def set_y_pred(self, y_pred):
        current_row = 0
        value = self.X._values()
        for i in range(len(self.lengths)):
            prev_y = [0.0] * self.ws
            for j in range(len(self.lengths[i])):
                start = current_row * self.num_col + 2 * self.ws + 1
                end = start + self.ws
                values[start:end] = prev_y
                prev_y.pop(0)
                prev_y.append(y_pred[current_row])
                current_row += 1


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

    if len(sys.argv) < 2:
        print('Usage: python3 torch_nn.py data_dir')
        exit()
    print(time.strftime('%Y-%m-%d %H:%M'))
    random.seed(42)
    data_dir = sys.argv[1]
    files = os.listdir(data_dir)
    indices = list(range(len(files)))
    random.shuffle(indices)

    train_files = [files[i] for i in indices[:int(len(indices * 0.80))]]
    val_files = [files[i] for i in indices[int(len(indices * 0.80)):]]
    train_Xes = []
    train_ys = []
    for fname in train_files:
        with open(os.path.join(data_dir, fname)) as f:
            train_Xes.append([])
            train_ys.append([])
            for line in f:
                line = line.strip()
                aa, b = line.split()
                aa = aa_to_index(aa)
                train_Xes[-1].append(aa)
                train_ys[-1].append(b)

    for fname in val_files:
        with open(os.path.join(data_dir, fname)) as f:
            val_Xes.append([])
            val_ys.append([])
            for line in f:
                line = line.strip()
                aa, b = line.split()
                aa = aa_to_index(aa)
                val_Xes[-1].append(aa)
                val_ys[-1].append(b)


    ws = 5
    train_dataset = Dataset(train_Xes, train_ys, ws)
    val_dataset = Dataset(val_Xes, val_ys, ws)
    print(time.strftime('%Y-%m-%d %H:%M'))

    num_layers = 8
    net = FeedForward(dataset.num_col, num_layers)

