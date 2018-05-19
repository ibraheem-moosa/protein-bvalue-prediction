import sys
if len(sys.argv) < 6:
    print('Usage: python3 preprocessdata.py input_directory num_of_proteins window_size X.txt y.txt')
    print('Example: python3 preprocessdata.py data 350 15 X.txt y.txt')
    exit()

import time
print(time.strftime('%Y-%m-%d %H:%M'))

import os
from Bio.PDB import Polypeptide
import numpy as np
import random
from collections import Counter

num_of_proteins = int(sys.argv[2])
files = random.sample(os.listdir(sys.argv[1]), num_of_proteins)
window_size = int(sys.argv[3])

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


X = []
y = []

for f in files:
    seq = []
    bfactors = []
    for line in open(os.path.join(sys.argv[1], f)):
        a, b = line.split()
        a = a.strip()
        b = float(b)
        seq.append(aa_to_index(a))
        bfactors.append(b)
    aa_counter = Counter(seq)
    aa_freq = [aa_counter[aa] / len(seq) for aa in range(21)] 
    windows = seq_to_windows(seq, window_size)
    input_data = windows
    # additional features
     
    protein_len = len(input_data)
    for i, data in enumerate(input_data):
        local_aa_counter = Counter(data)
        local_aa_freq = [local_aa_counter[aa] / len(data) for aa in range(21)] 
        pos = i / protein_len
        # relative position
        data.append(pos)
        # global amino acid frequency
        data.extend(aa_freq)
        # amino acid frequencey in the window
        data.extend(local_aa_freq)
    X.extend(input_data)
    bfactors = np.array(bfactors)
    bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
    y.extend(bfactors)


# function for oneshot representation
onehot = lambda x : ''.join(['1' if i == x else '0' for i in range(21)])
# function for binary representation
five_len_bin = lambda br: '0' * (5 - len(br)) + br

# write X
with open(sys.argv[4], 'w') as f:
    for i, x in enumerate(X):
        to_write = ''
        
        for i in range(2 * window_size + 1):
            #to_write += five_len_bin(bin(_x)[2:])
            to_write += onehot(x[i])
        to_write = ' '.join(list(to_write))
        
        for _x in x[2 * window_size + 1:]:
            to_write += ' ' + str(_x)
        f.write(to_write + '\n')

# write y
with open(sys.argv[5], 'w') as f:
    f.writelines(map(lambda _y: str(_y) + '\n', y))


# TODO Use OneHotEncoder which returns sparse matrix
#       Save to disk this sparse matrix using numpy function
#       Then load it directly later.
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#X = ohe.fit_transform(X)
#print(X.shape)
#y = np.array(y)


