from Bio.PDB import Polypeptide
import numpy as np
import random
from collections import Counter
import scipy.sparse as scsp

def aa_to_index(aa):
    """
    :param aa: Three character amino acid name.
    :returns: Integer index as per BioPython, unknown/non-standard amino acids return 20.
    """
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20

def get_window_at_pos(seq, pos, window_size):
    left_padding = max((window_size - pos), 0)
    right_padding = max(window_size - (len(seq) - pos - 1), 0)
    return [20] * left_padding + seq[pos-window_size+left_padding:pos] + seq[pos:pos+window_size+1-right_padding] + [20] * right_padding

def seq_to_windows(seq, window_size):
    """
    Split the protein sequence into a bunch of windows.
    :param seq: Protein sequence where each amino acid is represeented by integer index.
    :param window_size: Number of amino acids to take on each side to form window. 
                        So actual window size will be 2 * window_size + 1.
    :returns: A list of 2 * window_size + 1 sized lists. Part of window that are outside of
                protein sequence are represented by 20.
    """
    return [get_window_at_pos(seq, i, window_size) for i in range(len(seq))]

def seq_to_freq(seq):
    freq = [0.0] * 21
    for aa in seq:
        freq[aa] += 1
    freq = [freq[i] / len(seq) for i in range(len(freq))]
    return freq

def seq_to_local_freq(seq, window_size):
    local_freqs = []
    for i in range(len(seq)):
        local_seq = get_window_at_pos(seq, i, window_size)
        local_freqs.append((seq_to_freq(local_seq)))
    return local_freqs


def protein_to_features(seq, ws, local_freq_ws):
    num_of_columns = 1 + 21 + 21 + (2 * ws + 1) * 21
    non_zero_elem_per_row = 1 + 21 + 21 + (2 * ws + 1)
    data = []
    indices = []
    indptr = [0]
    windows = seq_to_windows(seq, ws)
    local_freqs = seq_to_local_freq(seq, local_freq_ws)
    global_freq = seq_to_freq(seq)
    for i in range(len(seq)):
        # relative position
        pos = i / len(seq)
        data.append(pos)
        indices.append(0)
        # global amino acid frequency
        for j in range(21):
            data.append(global_freq[j])
            indices.append(1 + j)
        # local amino acid frequency
        for j in range(21):
            data.append(local_freqs[i][j])
            indices.append(1 + 21 + j)
        # amino acids in window in one hot representation
        for j in range(len(windows[i])):
            data.append(1)
            indices.append(1 + 21 + 21 + j * 21 + windows[i][j])
        indptr.append(indptr[-1] +  non_zero_elem_per_row)
    return scsp.csr_matrix((data, indices, indptr), dtype=np.float64, shape = (len(seq), num_of_columns))

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 6:
        print('Usage: python3 preprocessdata.py input_directory num_of_proteins window_size local_freq_window_size X.txt [y.txt]')
        print('Example: python3 preprocessdata.py filtered_data 3484 9 18 X_9_18 y_9_18')
        exit()

    if len(sys.argv) > 5:
        target_available = True
    import time
    print(time.strftime('%Y-%m-%d %H:%M'))

    import os

    num_of_proteins = int(sys.argv[2])
    print('Total files: {}'.format(len(os.listdir(sys.argv[1]))))
    files = os.listdir(sys.argv[1])
    if num_of_proteins < len(files):
        files = random.sample(files, num_of_proteins)

    window_size = int(sys.argv[3])
    local_freq_ws = int(sys.argv[4])

    if target_available:
        y = []
    currently_processing = 0
    num_of_columns = 1 + 21 + 21 + (2 * window_size + 1) * 21
    non_zero_elem_per_row = 1 + 21 + 21 + (2 * window_size + 1)

    X = scsp.csr_matrix((0, num_of_columns))
    for f in files:
        print(f)
        if currently_processing % 100 == 0:
            print('Currently processing: {}'.format(currently_processing))
        currently_processing += 1
        seq = []
        if target_available:
            bfactors = []
        for line in open(os.path.join(sys.argv[1], f)):
            line = line.split()
            a = line[0].strip()
            seq.append(aa_to_index(a))
            if target_available:
                b = float(line[1])
                bfactors.append(b)
        X = scsp.vstack([X, protein_to_features(seq, window_size, local_freq_ws)])
        if target_available:
            #bfactors = np.log(np.array(bfactors))
            #bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
            y.extend(bfactors)

    # write X
    scsp.save_npz(sys.argv[5], X)
    # write y
    if target_available:
        y = np.array(y, dtype=np.float64)
        np.savez_compressed(sys.argv[6], y=y)
