from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.drop1 = nn.Dropout(0.2)
            self.drop2 = nn.Dropout(0.5)
            self.lkrelu = nn.LeakyReLU(0.01)
            self.layer1 = nn.Dense(256, dtype='float32')
            self.layer2 = nn.Dense(256, dtype='float32')
            self.layer3 = nn.Dense(256, dtype='float32')
            self.layer4 = nn.Dense(256, dtype='float32')
            self.layer5 = nn.Dense(256, dtype='float32')
            self.layer6 = nn.Dense(256, dtype='float32')
            self.layer7 = nn.Dense(256, dtype='float32')
            self.dense = nn.Dense(1, dtype='float32')

    def forward(self, x):
        y = self.lkrelu(self.layer1(x))
        y = self.lkrelu(self.layer2(x))
        y = self.lkrelu(self.layer3(x))
        y = self.lkrelu(self.layer4(x))
        y = self.lkrelu(self.layer5(x))
        y = self.lkrelu(self.layer6(x))
        y = self.lkrelu(self.layer7(x))
        return self.dense(y)

from bisect import bisect
class ProteinDataset(mx.gluon.data.Dataset):

    def __init__(self, X_files, y_files):
        self.X_files = X_files
        self.y_files = y_files
        self._protein_lengths = []
        for xf in X_files:
            X = nd.load(xf)[0]
            self._protein_lengths.append(X.shape[0])
        for i in range(1, len(self._protein_lengths)):
            self._protein_lengths[i] += self._protein_lengths[i-1]
        self._last_loaded_matrix = -1

    def __getitem__(self, idx):
        i = bisect(self._protein_lengths, idx)
        if i != self._last_loaded_matrix:
            self._X_cache = nd.load(self.X_files[i])[0]
            self._y_cache = nd.load(self.y_files[i])[0]
            self._last_loaded_matrix = i
        idx = idx - (0 if i == 0 else self._protein_lengths[i-1])
        return self._X_cache[idx], self._y_cache[idx]

    def __len__(self):
        return self._protein_lengths[-1]




import time
print(time.strftime('%Y-%m-%d %H:%M'))

import os
import sys
import numpy as np
import scipy.sparse as scsp

if len(sys.argv) < 3:
    print('Usage: python3 regression.py data_dir X_files y_files [seed]')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
    exit()

X_files_list = open(sys.argv[2])
X_files = []
for line in X_files_list:
    X_files.append(os.path.join(sys.argv[1], line[:-1]))
y_files_list = open(sys.argv[3])
y_files = []
for line in y_files_list:
    y_files.append(os.path.join(sys.argv[1], line[:-1]))

dataset = ProteinDataset(X_files, y_files)
print('Dataset init done ', len(dataset))
print(time.strftime('%Y-%m-%d %H:%M'))
batch_size = 256
dataloader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size, sampler=mx.gluon.data.SequentialSampler(len(dataset)))

net = MLP()
net.initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))

from mxnet import autograd
from mxnet import optimizer
from mxnet import gluon
optim = optimizer.Adam(learning_rate=0.01)
trainer = gluon.Trainer(net.collect_params(), optim)
for epoch in range(200):
    epoch_loss = 0
    batch_num = 0
    for X_batch, y_batch in dataloader:
        batch_num += 1
        with autograd.record():
            y_pred = net(X_batch)
            L = gluon.loss.L2Loss()
            loss = L(y_pred, y_batch)
        loss.backward()
        trainer.step(batch_size, ignore_stale_grad=True)
        epoch_loss += loss.mean().asscalar()
        #if batch_num % 500 == 0:
        #    print(epoch_loss)
        #print(epoch_loss)
        #print(loss.mean().asscalar())
    print(epoch,':',epoch_loss)
    print(time.strftime('%Y-%m-%d %H:%M'))
    if epoch % 10 == 0:
        print(net.dense.weight.data())
save_net = input('Do you want to save the net?')
if 'y' in save_net:
    fname = input('File Name;')
    net.save_params(fname)

#dataiter = mx.io.NDArrayIter(X, y, batch_size=32, last_batch_handle='discard')
#data = dataiter.next().data[0]
#label = dataiter.next().label[0].asnumpy()
#y_pred = net.forward(data).asnumpy()
#y_pred = y_pred.flatten()
#print(y_pred.shape)
#print(label.shape)

