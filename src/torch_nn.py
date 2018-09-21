import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import numpy as np
import torch.utils.data
import scipy.sparse as scsp
from bisect import bisect
import pickle

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


class ProteinDataset(torch.utils.data.Dataset):

    def __init__(self, X_files, y_files, cache_dir, limit=128*1024*1024, cache_prefix='cache-', validation=False):
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
