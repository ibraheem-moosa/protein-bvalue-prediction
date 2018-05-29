import time
print(time.strftime('%Y-%m-%d %H:%M'))

import sys
import os
from Bio.PDB import Polypeptide
import numpy as np

if len(sys.argv) < 2:
    print('Usage: python3 regression.py input_directory window_size')
    exit()

files = os.listdir(sys.argv[1])
window_size = int(sys.argv[2])

X = []
y = []

def seq_to_windows(seq, window_size):
    windows = [['NSA'] * (window_size - i) + seq[:i + window_size + 1] for i in range(window_size)]
    windows.extend([seq[i-window_size:i+window_size+1] for i in range(window_size, len(seq)-window_size)])
    windows.extend([seq[i - window_size:] + ['NSA'] * (i + window_size - len(seq) + 1) for i in range(len(seq) - window_size, len(seq))])
    return windows

for f in files:
    seq = []
    bfactors = []
    for line in open(os.path.join(sys.argv[1], f)):
        a, b = line.split()
        a = a.strip()
        b = float(b)
        seq.append(a)
        bfactors.append(b)
    X.extend(seq_to_windows(seq, window_size))
    bfactors = np.array(bfactors)
    bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
    y.extend(bfactors)

def aa_to_index(aa):
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20

X = list(map(lambda window: list(map(lambda aa: aa_to_index(aa), window)), X))


print(X[9])
import pandas as pd
X = pd.DataFrame(data=X)
print(X.shape)
print(X[:10])
import category_encoders as ce
bin_encoder = ce.BinaryEncoder(verbose=5, return_df=False, cols=range(X.shape[-1]))
bin_encoder.fit(X[:1000])
print('Fit done')
X = bin_encoder.transform(X)
#X = np.ndarray(X)
print(X.shape)
print(X[9])
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(sparse=False, n_values=21, dtype=np.float32)
#X = ohe.fit_transform(X)
y = np.array(y, dtype=np.float32)
y = np.reshape(y, (-1,1))
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)


import torch
import torch.utils.data
import scipy.sparse as sp

def to_torch_sparse_tensor(M):
    """Convert Scipy sparse matrix to torch sparse tensor."""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_pytorch_dataloader(X, y, batch_size=32):
    #tensor_X = torch.stack([torch.Tensor(i) for i in X])
    if sp.issparse(X):
        tensor_X = to_torch_sparse_tensor(X)
    else:
        tensor_X = torch.tensor(X, dtype=torch.float32)
    #tensor_y = torch.stack([torch.Tensor(i) for i in y])
    tensor_y = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(
                        tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size)
    return dataloader


trainloader = to_pytorch_dataloader(X_train, y_train)
testloader = to_pytorch_dataloader(X_test, y_test, batch_size=1)
# data input and preprocessing done


import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.functional import dropout

class FeedForward(nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        # input is 21 * (2 * 2 + 1) = 105
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = dropout(relu(self.fc1(x)))
        x = dropout(relu(self.fc2(x)))
        x = dropout(relu(self.fc3(x)))
        x = dropout(relu(self.fc4(x)))
        x = self.fc5(x)
        return x

net = FeedForward(len(X[0]))

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), 
                        lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, true_outputs = data
        optimizer.zero_grad()
        predicted_outputs = net(inputs)
        loss = criterion(predicted_outputs, 
                                true_outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch: {} done. Loss: {}'.format(
                            epoch, running_loss))

print('Training Done')


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

