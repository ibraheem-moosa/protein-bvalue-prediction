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
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, 1)
        self._init_weights_()

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = relu(self.fc5(x))
        x = relu(self.fc6(x))
        x = relu(self.fc7(x))
        x = self.fc8(x)
        return x
    
    def _init_weights_(self):
        gain = nn.init.calculate_gain('relu')
        gain = 10
        bias_std = 1
        weight_std = 10
        init_method = nn.init.normal_
        bias_init_method = nn.init.normal_
        init_method(self.fc1.weight.data, std=weight_std)
        init_method(self.fc2.weight.data, std=weight_std)
        init_method(self.fc3.weight.data, std=weight_std)
        init_method(self.fc4.weight.data, std=weight_std)
        init_method(self.fc5.weight.data, std=weight_std)
        init_method(self.fc6.weight.data, std=weight_std)
        init_method(self.fc7.weight.data, std=weight_std)
        init_method(self.fc8.weight.data, std=weight_std)
        bias_init_method(self.fc1.bias.data, std=bias_std)
        bias_init_method(self.fc2.bias.data, std=bias_std)
        bias_init_method(self.fc3.bias.data, std=bias_std)
        bias_init_method(self.fc4.bias.data, std=bias_std)
        bias_init_method(self.fc5.bias.data, std=bias_std)
        bias_init_method(self.fc6.bias.data, std=bias_std)
        bias_init_method(self.fc7.bias.data, std=bias_std)
        bias_init_method(self.fc8.bias.data, std=bias_std)


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
            assert(xf[32:] == yf[32:])
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

    if len(sys.argv) == 12:
        warm_start_model_params = torch.load(sys.argv[10])
        warm_start_last_epoch = int(sys.argv[11])
    else:
        warm_start_model_params = None
        warm_start_last_epoch = -1

    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
            sampler=torch.utils.data.sampler.SequentialSampler(dataset))

    net = FeedForward(dataset._num_of_features)
    if warm_start_model_params != None:
        net.load_state_dict(warm_start_model_params)

    from math import sqrt
    import torch.optim as optim
    init_lr = 0.5
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params' : net.parameters(), 'initial_lr' : init_lr}],
                            lr=init_lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1.00, last_epoch=warm_start_last_epoch)
    
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    for epoch in range(warm_start_last_epoch + 1, warm_start_last_epoch + 1 + 500):
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

        train_pred, train_true = net.predict(train_dataset)
        train_pcc = list(map(lambda t: pearsonr(t[0], t[1])[0], zip(train_pred, train_true)))
        train_pcc = sum(train_pcc) / len(train_pcc)
        train_r2 = list(map(lambda t: r2_score(t[1], t[0]), zip(train_pred, train_true)))
        train_r2 = sum(train_r2) / len(train_r2)
        net.train()
        '''
        R2 = 1 - (mean squared error) / (Variance of truth)
        '''
        print('-----------------------------------------------')
        print('Validation Prediction min, max, mean, std: {}'.format(summarize_ndarrays(validation_pred)))
        print('Validation True min, max, mean, std: {}'.format(summarize_ndarrays(validation_true)))
        print('Validation PCC: {}'.format(validation_pcc))
        print('Validation R2: {}'.format(validation_r2))
        print('-----------------------------------------------')
        print('Train Prediction min, max, mean, std: {}'.format(summarize_ndarrays(train_pred)))
        print('Train True min, max, mean, std: {}'.format(summarize_ndarrays(train_true)))
        print('Train PCC: {}'.format(train_pcc))
        print('Train R2: {}'.format(train_r2))
        print('-----------------------------------------------')
        for name, param in net.named_parameters():
            print(name)
            print(summarize_tensor(param))
        print('###############################################')
        print(time.strftime('%Y-%m-%d %H:%M'))
    #    for g in optimizer.param_groups:
    #        g['lr'] = init_lr / sqrt(epoch + 1)
        #if (epoch + 1) % 5 == 0:
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

