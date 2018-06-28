import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect

class FeedForward(nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 1)
        self._init_weights_()

    def forward(self, x):
        x = dropout(leaky_relu(self.fc1(x)),p=0.2,inplace=True)
        x = dropout(leaky_relu(self.fc2(x)),inplace=True)
        x = dropout(leaky_relu(self.fc3(x)),inplace=True)
        x = dropout(leaky_relu(self.fc4(x)),inplace=True)
        x = dropout(leaky_relu(self.fc5(x)),inplace=True)
        x = dropout(leaky_relu(self.fc6(x)),inplace=True)
        x = dropout(leaky_relu(self.fc7(x)),inplace=True)
        x = self.fc8(x)
        return x
    
    def _init_weights_(self):
        init_method = nn.init.xavier_normal_
        init_method(self.fc1.weight.data)
        init_method(self.fc2.weight.data)
        init_method(self.fc3.weight.data)
        init_method(self.fc4.weight.data)
        init_method(self.fc5.weight.data)
        init_method(self.fc6.weight.data)
        init_method(self.fc7.weight.data)
        init_method(self.fc8.weight.data)

    def predict(self, dataset):
        with torch.no_grad():
            Y_pred = []
            Y_true = []
            for x, y in dataset:
                y_pred = self.forward(x).numpy().flatten()
                Y_pred.append(y_pred)
                Y_true.append(y.numpy().flatten())
            return np.hstack(Y_pred), np.hstack(Y_true)

class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files, cache_dir, limit=4*1024*1024, cache_prefix=None):
        self.cache_dir = cache_dir
        self._protein_lengths = []
        if cache_prefix == None:
            X_cache_prefix = 'X_cache-'
            y_cache_prefix = 'y_cache-'
        else:
            X_cache_prefix = 'X_' + cache_prefix
            y_cache_prefix = 'y_' + cache_prefix

        loaded_Xes = []
        loaded_ys = []
        current_size = 0
        matrices_saved = 0
        total_protein_length = 0
        for xf,yf in zip(X_files, y_files):
            X = scsp.load_npz(xf)
            y = np.load(yf)['y']
            assert(X.shape[0] == len(y))
            total_protein_length += X.shape[0]
            loaded_Xes.append(X)
            loaded_ys.append(y)
            current_size += X.shape[0] * X.shape[1]
            if current_size >= limit:
                scsp.save_npz(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), scsp.vstack(loaded_Xes))
                np.savez(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), y=np.hstack(loaded_ys))
                self._protein_lengths.append(total_protein_length)
                loaded_Xes = []
                loaded_ys = []
                matrices_saved += 1
                current_size = 0
            self._num_of_features = X.shape[1]
        scsp.save_npz(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), scsp.vstack(loaded_Xes))
        np.savez(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), y=np.hstack(loaded_ys))
        self._protein_lengths.append(total_protein_length)
        print(self._protein_lengths)
        self._last_loaded_matrix = -1
        self._X_cache_prefix = X_cache_prefix
        self._y_cache_prefix = y_cache_prefix

    def __getitem__(self, idx):
        i = bisect(self._protein_lengths, idx)
        if i != self._last_loaded_matrix:
            self._X_cache = scsp.load_npz(os.path.join(self.cache_dir, self._X_cache_prefix + str(i) + '.npz')).toarray()
            self._y_cache = np.load(os.path.join(self.cache_dir, self._y_cache_prefix + str(i) + '.npz'))['y']
            self._last_loaded_matrix = i
            print(i)
        idx = idx - (0 if i == 0 else self._protein_lengths[i-1])
        retX = self._X_cache[idx]
        rety = self._y_cache[idx]
        return retX, rety

    def __len__(self):
        return self._protein_lengths[-1]

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 9:
        print('Usage: python3 regression.py data_dir X_files y_files validation_dir X_validation_files y_validation_files cache_dir limit checkpoint_dir [warm_start_model] [warm_start_epoch]')
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
    dataset = ProteinDataset(X_files, y_files, sys.argv[7], int(sys.argv[8]))
    print('Dataset init done ', len(dataset))
    validation_dataset = ProteinDataset(X_validation_files, y_validation_files, sys.argv[7], int(sys.argv[8]), 'validation-')
    print('Validation dataset init done ', len(validation_dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))

    if len(sys.argv) == 12:
        warm_start_model_params = torch.load(sys.argv[10])
        warm_start_last_epoch = int(sys.argv[11])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
            sampler=torch.utils.data.sampler.SequentialSampler(dataset))
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=10 * batch_size, 
            sampler=torch.utils.data.sampler.SequentialSampler(validation_dataset))


    net = FeedForward(dataset._num_of_features)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    from math import sqrt
    import torch.optim as optim
    init_lr = 0.002
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99, last_epoch=warm_start_last_epoch)
    
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 500):
        scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
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
                                epoch, running_loss))
        print(time.strftime('%Y-%m-%d %H:%M'))
        validation_pred, validation_true = net.predict(validation_dataloader)
        train_pred, train_true = net.predict(dataloader)
        '''
        R2 = 1 - (mean squared error) / (Variance of truth)
        '''
        print('Validation PCC: {}'.format(pearsonr(validation_pred, validation_true)[0]))
        print('Validation R2: {}'.format(r2_score(validation_true, validation_pred)))
        print('Train PCC: {}'.format(pearsonr(train_pred, train_true)[0]))
        print('Train R2: {}'.format(r2_score(train_true, train_pred)))
        print(time.strftime('%Y-%m-%d %H:%M'))
    #    for g in optimizer.param_groups:
    #        g['lr'] = init_lr / sqrt(epoch + 1)
        if (epoch + 1) % 20 == 0:
            torch.save(net.state_dict(), os.path.join(sys.argv[9], 'net-{0:02d}'.format(epoch)))
        random.shuffle(indices)
        X_files = [X_files[i] for i in indices]
        y_files = [y_files[i] for i in indices]
        dataset = ProteinDataset(X_files, y_files, sys.argv[7], int(sys.argv[8]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                     sampler=torch.utils.data.sampler.SequentialSampler(dataset))

    save_net = input('Do you want to save the net?')
    if 'y' in save_net:
        fname = input('File Name;')
        torch.save(net.state_dict(), fname)

