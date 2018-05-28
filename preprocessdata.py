import sys
if len(sys.argv) < 6:
    print('Usage: python3 preprocessdata.py input_directory num_of_proteins window_size local_freq_window_size X.txt y.txt')
    print('Example: python3 preprocessdata.py data 350 15 30 X.txt y.txt')
    exit()

import time
print(time.strftime('%Y-%m-%d %H:%M'))

import os
from Bio.PDB import Polypeptide
import numpy as np
import random
from collections import Counter

num_of_proteins = int(sys.argv[2])
print('Total files: {}'.format(len(os.listdir(sys.argv[1]))))
files = random.sample(os.listdir(sys.argv[1]), num_of_proteins)
window_size = int(sys.argv[3])
local_freq_ws = int(sys.argv[4])

def aa_to_index(aa):
    """
    :param aa: Three character amino acid name.
    :returns: Integer index as per BioPython, unknown/non-standard amino acids return 20.
    """
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20

def seq_to_windows(seq, window_size):
    """
    Split the protein sequence into a bunch of windows.
    :param seq: Protein sequence where each amino acid is represeented by integer index.
    :param window_size: Number of amino acids to take on each side to form window. 
                        So actual window size will be 2 * window_size + 1.
    :returns: A list of 2 * window_size + 1 sized lists. Part of window that are outside of
                protein sequence are represented by 20.
    """
    windows = [[20] * (window_size - i) + seq[:i + window_size + 1] for i in range(window_size)]
    windows.extend([seq[i-window_size:i+window_size+1] for i in range(window_size, len(seq)-window_size)])
    windows.extend([seq[i - window_size:] + [20] * (i + window_size - len(seq) + 1) for i in range(len(seq) - window_size, len(seq))])
    return windows

def seq_to_freq(seq):
    freq = [0.0] * 21
    for aa in seq:
        freq[aa] += 1
    freq = [freq[i] / len(seq) for i in range(len(freq))]
    return freq

def seq_to_local_freq(seq, window_size):
    local_freqs = []
    for i in range(len(seq)):
        if i < window_size:
            local_seq = [20] * (window_size - i) + seq[:i + window_size + 1]
        elif i + window_size > len(seq) - 1:
            local_seq = seq[i - window_size:] + [20] * (i + window_size - len(seq) + 1)
        else:
            local_seq = seq[i - window_size: i + window_size + 1]
        local_freqs.append((seq_to_freq(local_seq)))
    return local_freqs


y = []
X_data = []
X_row = []
X_col = []

currently_processing = 0
current_row = 0
num_of_columns = 1 + 21 + 21 + (2 * window_size + 1) * 21
for f in files:
    if currently_processing % 10 == 0:
        print('Currently processing: {}'.format(currently_processing))
    currently_processing += 1
    seq = []
    bfactors = []
    for line in open(os.path.join(sys.argv[1], f)):
        a, b = line.split()
        a = a.strip()
        b = float(b)
        seq.append(aa_to_index(a))
        bfactors.append(b)
    windows = seq_to_windows(seq, window_size)
    # additional features
    local_freqs = seq_to_local_freq(seq, local_freq_ws)
    global_freq = seq_to_freq(seq)
    for i in range(len(seq)):
        # relative position
        pos = i / len(seq)
        X_data.append(pos)
        X_row.append(current_row)
        X_col.append(0)
        # global amino acid frequency
        for j in range(21):
            X_data.append(global_freq[j])
            X_row.append(current_row)
            X_col.append(1 + j)
        # local amino acid frequency
        for j in range(21):
            X_data.append(local_freqs[i][j])
            X_row.append(current_row)
            X_col.append(1 + 21 + j)
        # amino acids in window in one hot representation
        for j in range(len(windows[i])):
            X_data.append(1)
            X_row.append(i)
            X_col.append(1 + 21 + 21 + j * 21 + windows[i][j])

        current_row += 1

    bfactors = np.log(np.array(bfactors))
    bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
    y.extend(bfactors)

from scipy.sparse import csr_matrix
X = csr_matrix((X_data, (X_row, X_col)), dtype=np.float32, shape=(current_row, num_of_columns))
y = np.array(y, dtype=np.float32)

# function for oneshot representation
onehot = lambda x : ''.join(['1' if i == x else '0' for i in range(21)])
# function for binary representation
five_len_bin = lambda br: '0' * (5 - len(br)) + br

# write X
'''
with open(sys.argv[5], 'w') as f:
    for i, x in enumerate(X):
        to_write = ''
        
        for i in range(2 * window_size + 1):
            #to_write += five_len_bin(bin(_x)[2:])
            to_write += onehot(x[i])
        to_write = ' '.join(list(to_write))
        
        for _x in x[2 * window_size + 1:]:
            to_write += ' ' + str(_x)
        f.write(to_write + '\n')
'''

from scipy.sparse import save_npz
save_npz(sys.argv[5], X)

# write y
'''
with open(sys.argv[6], 'w') as f:
    f.writelines(map(lambda _y: str(_y) + '\n', y))
'''
np.savez_compressed(sys.argv[6], y=y)

# TODO Use OneHotEncoder which returns sparse matrix
#       Save to disk this sparse matrix using numpy function
#       Then load it directly later.
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#X = ohe.fit_transform(X)
#print(X.shape)
#y = np.array(y)


