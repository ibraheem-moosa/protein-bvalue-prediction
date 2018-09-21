import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np

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
        criterion = nn.MSELoss()
        for i in range(num_iter):
            optimizer = optim.Adam([{'params' : self.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, weight_decay=0.1)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
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
                index[0].extend([1] * (3 * ws + 1))
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

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 9:
        print('Usage: python3 regression.py data_dir X_files y_files validation_dir X_validation_files y_validation_files cache_dir checkpoint_dir [warm_start_model] [warm_start_epoch]')
        print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
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
    assert(len(X_files) == len(y_files))
    '''
    X_validation_files_list = open(sys.argv[5])
    X_validation_files = []
    for line in X_validation_files_list:
        X_validation_files.append(os.path.join(sys.argv[4], line[:-1]))
    y_validation_files_list = open(sys.argv[6])
    y_validation_files = []
    for line in y_validation_files_list:
        y_validation_files.append(os.path.join(sys.argv[4], line[:-1]))
    '''

    import random
    random.seed(42)
    indices = list(range(len(X_files)))
    random.shuffle(indices)

    X_files = [X_files[i] for i in indices]
    y_files = [y_files[i] for i in indices]
    X_validation_files = X_files[-len(indices) // 5:]
    y_validation_files = y_files[-len(indices) // 5:]
    X_files = X_files[:-len(indices) // 5]
    y_files = y_files[:-len(indices) // 5]
    indices = list(range(len(X_files)))
    dataset = ProteinDataset(X_files, y_files, sys.argv[7], int(sys.argv[8]))
    train_dataset = ProteinDataset(X_files, y_files, sys.argv[7], limit=int(sys.argv[8]), cache_prefix='train-', validation=True)
    print('Dataset init done ', len(dataset))
    validation_dataset = ProteinDataset(X_validation_files, y_validation_files, sys.argv[7], limit=int(sys.argv[8]), cache_prefix='validation-', validation=True)
    print('Validation dataset init done ', len(validation_dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))

    if len(sys.argv) == 11:
        warm_start_model_params = torch.load(sys.argv[9])
        warm_start_last_epoch = int(sys.argv[10])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
            sampler=torch.utils.data.sampler.SequentialSampler(dataset))

    print(dataset._num_of_features)
    net = FeedForward(dataset._num_of_features)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    from math import sqrt
    import torch.optim as optim
    init_lr = 0.0005
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=warm_start_last_epoch)
    
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    import math
    best_validation_pcc = -math.inf
    best_validation_pcc_found_at = -1

    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 200):
        print('###############################################')
        scheduler.step()
        running_loss = 0.0
        update_count = 0
        for i, data in enumerate(dataloader):
            update_count += 1
            inputs, true_outputs = data
            true_outputs = true_outputs.view(true_outputs.shape[0], 1)
            optimizer.zero_grad()
            predicted_outputs = net(inputs)
            loss = criterion(predicted_outputs, 
                                    true_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch: {} done. Loss: {}'.format(
                                epoch, running_loss / update_count))
        print(time.strftime('%Y-%m-%d %H:%M'))
        net.eval()
        validation_pred, validation_true = net.predict(validation_dataset)
        validation_pcc = list(map(lambda t: pearsonr(t[0], t[1])[0], zip(validation_pred, validation_true)))
        validation_pcc = sum(validation_pcc) / len(validation_pcc)
        validation_r2 = list(map(lambda t: r2_score(t[1], t[0]), zip(validation_pred, validation_true)))
        validation_r2 = sum(validation_r2) / len(validation_r2)

        if validation_pcc > best_validation_pcc:
            best_validation_pcc = validation_pcc
            best_validation_pcc_found_at = epoch
        '''
        train_pred, train_true = net.predict(train_dataset)
        train_pcc = list(map(lambda t: pearsonr(t[0], t[1])[0], zip(train_pred, train_true)))
        train_pcc = sum(train_pcc) / len(train_pcc)
        train_r2 = list(map(lambda t: r2_score(t[1], t[0]), zip(train_pred, train_true)))
        train_r2 = sum(train_r2) / len(train_r2)
        '''
        #R2 = 1 - (mean squared error) / (Variance of truth)
        print('-----------------------------------------------')
        print('Validation Prediction min, max, mean, std: {}'.format(summarize_ndarrays(validation_pred)))
        print('Validation True min, max, mean, std: {}'.format(summarize_ndarrays(validation_true)))
        print('Validation PCC: {}'.format(validation_pcc))
        print('Validation R2: {}'.format(validation_r2))
        print('-----------------------------------------------')
        '''
        print('Train Prediction min, max, mean, std: {}'.format(summarize_ndarrays(train_pred)))
        print('Train True min, max, mean, std: {}'.format(summarize_ndarrays(train_true)))
        print('Train PCC: {}'.format(train_pcc))
        print('Train R2: {}'.format(train_r2))
        print('-----------------------------------------------')
        '''
        for name, param in net.named_parameters():
            print(name)
            print(summarize_tensor(param))
        print('###############################################')
        print(time.strftime('%Y-%m-%d %H:%M'))
    #    for g in optimizer.param_groups:
    #        g['lr'] = init_lr / sqrt(epoch + 1)
        #if (epoch + 1) % 5 == 0:
        torch.save(net.state_dict(), os.path.join(sys.argv[8], 'net-{0:02d}'.format(epoch)))
        if epoch - best_validation_pcc_found_at >= 10:
            break
        net.train()
        random.shuffle(indices)
        X_files = [X_files[i] for i in indices]
        y_files = [y_files[i] for i in indices]
        dataset = ProteinDataset(X_files, y_files, sys.argv[7], int(sys.argv[8]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                     sampler=torch.utils.data.sampler.SequentialSampler(dataset))
    '''
    save_net = input('Do you want to save the net?')
    if 'y' in save_net:
        fname = input('File Name;')
        torch.save(net.state_dict(), fname)
    '''
