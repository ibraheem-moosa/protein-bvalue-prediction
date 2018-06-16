import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import dropout

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
        x = dropout(relu(self.fc1(x)),p=0.2,inplace=True)
        x = dropout(relu(self.fc2(x)),inplace=True)
        x = dropout(relu(self.fc3(x)),inplace=True)
        x = dropout(relu(self.fc4(x)),inplace=True)
        x = dropout(relu(self.fc5(x)),inplace=True)
        x = dropout(relu(self.fc6(x)),inplace=True)
        x = dropout(relu(self.fc7(x)),inplace=True)
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

import torch.utils.data
import scipy.sparse as scsp
import numpy as np
from bisect import bisect

class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files, cache_dir, limit):
        self.cache_dir = cache_dir
        self._protein_lengths = []
        loaded_Xes = []
        loaded_ys = []
        current_size = 0
        matrices_saved = 0
        total_protein_length = 0
        for xf,yf in zip(X_files, y_files):
            X = scsp.load_npz(xf)
            y = np.load(yf)['y']
            total_protein_length += X.shape[0]
            loaded_Xes.append(X)
            loaded_ys.append(y)
            current_size += X.shape[0] * X.shape[1]
            if current_size >= limit:
                scsp.save_npz(os.path.join(cache_dir, 'X_cache-' + str(matrices_saved)), scsp.vstack(loaded_Xes))
                np.savez(os.path.join(cache_dir, 'y_cache-' + str(matrices_saved)), y=np.hstack(loaded_ys))
                self._protein_lengths.append(total_protein_length)
                loaded_Xes = []
                loaded_ys = []
                matrices_saved += 1
                current_size = 0
            self._num_of_features = X.shape[1]
        scsp.save_npz(os.path.join(cache_dir, 'X_cache-' + str(matrices_saved)), scsp.vstack(loaded_Xes))
        np.savez(os.path.join(cache_dir, 'y_cache-' + str(matrices_saved)), y=np.hstack(loaded_ys))
        self._protein_lengths.append(total_protein_length)
        print(self._num_of_features)
        self._last_loaded_matrix = -1

    def __getitem__(self, idx):
        i = bisect(self._protein_lengths, idx)
        if i != self._last_loaded_matrix:
            self._X_cache = scsp.load_npz(os.path.join(self.cache_dir, 'X_cache-' + str(i) + '.npz')).toarray()
            self._y_cache = np.load(os.path.join(self.cache_dir, 'y_cache-' + str(i) + '.npz'))['y']
            self._last_loaded_matrix = i
        idx = idx - (0 if i == 0 else self._protein_lengths[i-1])
        return self._X_cache[idx], self._y_cache[idx]

    def __len__(self):
        return self._protein_lengths[-1]


import sys
import numpy as np

if len(sys.argv) < 3:
    print('Usage: python3 regression.py data_dir X_files y_files cache_dir limit [seed]')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
    exit()

import time
print(time.strftime('%Y-%m-%d %H:%M'))

if len(sys.argv) == 7:
    seed = int(sys.argv[6])
else:
    seed = None

import os
X_files_list = open(sys.argv[2])
X_files = []
for line in X_files_list:
    X_files.append(os.path.join(sys.argv[1], line[:-1]))
y_files_list = open(sys.argv[3])
y_files = []
for line in y_files_list:
    y_files.append(os.path.join(sys.argv[1], line[:-1]))

dataset = ProteinDataset(X_files, y_files, sys.argv[4], int(sys.argv[5]))
print('Dataset init done ', len(dataset))
print(time.strftime('%Y-%m-%d %H:%M'))

batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        sampler=torch.utils.data.sampler.SequentialSampler(dataset))

net = FeedForward(dataset._num_of_features)

from math import sqrt
import torch.optim as optim
init_lr = 0.0001
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), 
                        lr=init_lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

for epoch in range(500):
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
#    for g in optimizer.param_groups:
#        g['lr'] = init_lr / sqrt(epoch + 1)


save_net = input('Do you want to save the net?')
if 'y' in save_net:
    fname = input('File Name;')
    torch.save(net.state_dict(), fname)


'''
true_y = []
pred_y = []
with torch.no_grad():
    for data in testloader:
        inputs, true_outputs = data
        predicted_outputs = net(inputs)
        true_y.extend(true_outputs)
        pred_y.extend(predicted_outputs)

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]

print('Test MSE: {}'.format(mse(true_y, pred_y)))
print('Test PCC: {}'.format(p(true_y, pred_y)))
'''
