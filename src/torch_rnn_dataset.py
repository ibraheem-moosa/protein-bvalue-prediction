import torch
from torch.nn.utils.rnn import pad_sequence
from numpy import load
from scipy.sparse import load_npz


class ProteinDataset:

    def __init__(self, files, batch_size):
        self._Xes = []
        self._yes = []
        for xf,yf in files:
            X = torch.from_numpy(load_npz(xf).toarray()).reshape((-1, 21))
            y = torch.from_numpy(load(yf)['y']).reshape((-1, 1))
            assert(X.shape[0] == y.shape[0])
            X, y = X.cuda(), y.cuda()
            self._Xes.append(X)
            self._yes.append(y)

        self._Xes.sort(key=lambda x:x.shape[0], reverse=True)
        self._yes.sort(key=lambda y:y.shape[0], reverse=True)
        self._lengths = [x.shape[0] for x in self._Xes]
        self._total_length = sum(self._lengths)

        X_batches = []
        y_batches = []
        lengths = []
        for i in range(0, len(files), batch_size):
            stride = min(len(files) - i, batch_size)
            lengths.append(self._lengths[i:i+stride])
            X_batches.append(
                        pad_sequence(self._Xes[i:i+stride], batch_first=True))
            y_batches.append(
                        pad_sequence(self._yes[i:i+stride], batch_first=True))

        self._Xes = X_batches
        self._yes = y_batches
        self._lengths = lengths

    def __getitem__(self, idx):
        return self._Xes[idx], self._yes[idx], self._lengths[idx]

    def __len__(self):
        return len(self._Xes)

    def total_length(self):
        return self._total_length

