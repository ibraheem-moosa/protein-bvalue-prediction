import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import tanh
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect
import pickle

class FeedForward(nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        '''
        self.fc1 = nn.Linear(input_size, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, 1)
        '''
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        '''
        x = dropout(leaky_relu(self.fc1(x)), p=0.2)
        x = dropout(leaky_relu(self.fc2(x)))
        x = dropout(leaky_relu(self.fc3(x)))
        x = dropout(leaky_relu(self.fc4(x)))
        x = dropout(leaky_relu(self.fc5(x)))
        x = dropout(leaky_relu(self.fc6(x)))
        x = dropout(leaky_relu(self.fc7(x)))
        x = self.fc8(x)
        '''
        x = self.fc1(x)
        return x
    
    def predict(self, dataset):
        with torch.no_grad():
            Y_pred = []
            Y_true = []
            for i in range(len(dataset)):
                #print(i)
                X, y = dataset[i]
                y_pred = self.forward(torch.from_numpy(X.toarray())).numpy().flatten()
                Y_pred.append(y_pred)
                Y_true.append(y)
            return Y_pred, Y_true


class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files, cache_dir, limit=4*1024*1024, cache_prefix='cache-', validation=False):
        self.cache_dir = cache_dir
        self._cache_element_counts = []
        self._validation = validation
        X_cache_prefix = 'X_' + cache_prefix
        y_cache_prefix = 'y_' + cache_prefix

        loaded_Xes = []
        loaded_ys = []
        current_size = 0
        matrices_saved = 0
        total_cache_element_count = 0
        for xf,yf in zip(X_files, y_files):
            X = scsp.load_npz(xf)
            y = np.load(yf)['y']
            assert(xf.split('/')[-1][1:] == yf.split('/')[-1][1:])
            assert(X.shape[0] == len(y))
            if validation:
                total_cache_element_count += 1
            else:
                total_cache_element_count += X.shape[0]
            loaded_Xes.append(X)
            loaded_ys.append(y)
            current_size += X.shape[0] * X.shape[1]
            if current_size >= limit:
                if validation:
                    f = open(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), 'wb')
                    pickle.dump(loaded_Xes, f)
                    f.close()
                    f = open(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), 'wb')
                    pickle.dump(loaded_ys, f)
                    f.close()
                else:
                    scsp.save_npz(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), scsp.vstack(loaded_Xes))
                    np.savez(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), y=np.hstack(loaded_ys))
                self._cache_element_counts.append(total_cache_element_count)
                loaded_Xes = []
                loaded_ys = []
                matrices_saved += 1
                current_size = 0
            self._num_of_features = X.shape[1]
        if validation:
            f = open(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), 'wb')
            pickle.dump(loaded_Xes, f)
            f.close()
            f = open(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), 'wb')
            pickle.dump(loaded_ys, f)
            f.close()
        else:
            scsp.save_npz(os.path.join(cache_dir, X_cache_prefix + str(matrices_saved)), scsp.vstack(loaded_Xes))
            np.savez(os.path.join(cache_dir, y_cache_prefix + str(matrices_saved)), y=np.hstack(loaded_ys))
        self._cache_element_counts.append(total_cache_element_count)
        self._last_loaded = -1
        self._X_cache_prefix = X_cache_prefix
        self._y_cache_prefix = y_cache_prefix

    def __getitem__(self, idx):
        i = bisect(self._cache_element_counts, idx)
        if i != self._last_loaded:
            if self._validation:
                with open(os.path.join(self.cache_dir, self._X_cache_prefix + str(i)), 'rb') as X_file:
                    self._X_cache = pickle.load(X_file)
                with open(os.path.join(self.cache_dir, self._y_cache_prefix + str(i)), 'rb') as y_file:
                    self._y_cache = pickle.load(y_file)
            else:
                self._X_cache = scsp.load_npz(os.path.join(self.cache_dir, self._X_cache_prefix + str(i) + '.npz')).toarray()
                self._y_cache = np.load(os.path.join(self.cache_dir, self._y_cache_prefix + str(i) + '.npz'))['y']
            self._last_loaded = i
        idx = idx - (0 if i == 0 else self._cache_element_counts[i-1])
        assert(len(self._X_cache) == len(self._y_cache))
        if(idx > len(self._X_cache)):
            print(i, idx)
        retX = self._X_cache[idx]
        rety = self._y_cache[idx]
        return retX, rety

    def __len__(self):
        return self._cache_element_counts[-1]

def summarize_tensor(tensor):
    return torch.min(tensor).item(), torch.max(tensor).item(), torch.mean(tensor).item(), torch.std(tensor).item()
def summarize_ndarray(array):
    return np.min(array), np.max(array), np.mean(array), np.std(array)
def summarize_ndarrays(arrays):
    mins, maxs, means, stds = tuple(zip(*list(map(summarize_ndarray, arrays))))
    return np.mean(mins), np.mean(maxs), np.mean(means), np.mean(stds)

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 7:
        print('Usage: python3 regression.py data_dir X_files y_files cache_dir limit models_dir')
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

    import random
    indices = list(range(len(X_files)))
    random.shuffle(indices)
    X_files = [X_files[i] for i in indices[:len(indices)]]
    y_files = [y_files[i] for i in indices[:len(indices)]]
    dataset = ProteinDataset(X_files, y_files, sys.argv[4], limit=int(sys.argv[5]), cache_prefix='ensemble-', validation=True)
    print('Dataset init done ', len(dataset))
    print(time.strftime('%Y-%m-%d %H:%M'))

    models = os.listdir(sys.argv[6])
    nets = []
    for i in range(len(models)):
        model_params = torch.load(os.path.join(sys.argv[6], models[i]))
        net = FeedForward(dataset._num_of_features)
        net.load_state_dict(model_params)
        net.eval()
        nets.append(net)

    print('Models loaded')
    nets_predictions = [[] for i in range(len(nets))]
    true_ys = []
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(i)
        X, y = dataset[i]
        true_ys.append(y)
        for j in range(len(nets)):
            with torch.no_grad():
                y_pred = nets[j].forward(torch.from_numpy(X.toarray())).numpy().flatten()
                nets_predictions[j].append(y_pred)

    print(time.strftime('%Y-%m-%d %H:%M'))
    print('Model predictions computed')
    print('Creating dataset for estimating model weights')
    X = [np.vstack([nets_predictions[j][i] for j in range(len(nets_predictions))]).transpose() for i in range(len(nets_predictions[0]))]
    X = np.vstack(X)
    y = np.hstack(true_ys)

    print('Try scipy nnls')
    from scipy.optimize import nnls
    model_weights_nn = nnls(X,y)
    print(model_weights_nn)
    model_weights_nn = model_weights_nn[0]
    print(len(model_weights_nn))
    indices = list(range(len(nets)))
    indices.sort(key = lambda i: model_weights_nn[i], reverse=True)
    non_zero_model_count = len(model_weights_nn) - list(model_weights_nn).count(0.0)
    indices = indices[:non_zero_model_count]
    print(indices)
    from scipy.stats import pearsonr
    models_pccs = [list(map(lambda t: pearsonr(t[0], t[1])[0], zip(true_ys, nets_predictions[i]))) for i in indices]
    models_mean_pccs = [sum(pccs) / len(pccs) for pccs in models_pccs]

    ensemble_predictions = [[model_weights_nn[indices[0]] * nets_predictions[indices[0]][j] for j in range(len(nets_predictions[indices[0]]))]]
    for i in indices[1:]:
        ensemble_predictions.append([ensemble_predictions[-1][j] + nets_predictions[i][j] * model_weights_nn[i] for j in range(len(nets_predictions[i]))])
    ensemble_pccs = [list(map(lambda t: pearsonr(t[0], t[1])[0], zip(true_ys, ensemble_predictions[i]))) for i in range(len(ensemble_predictions))]
    ensemble_mean_pccs = [sum(pccs) / len(pccs) for pccs in ensemble_pccs]

    import matplotlib.pyplot as plt
    plt.plot(ensemble_mean_pccs, 'b-')
    plt.plot(models_mean_pccs, 'g-')
    plt.show()
    print('Best ensemble pcc: {}'.format(max(ensemble_mean_pccs)))
