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

def get_window_at_pos(seq, pos, window_size, padding=True):
    left_padding = max((window_size - pos), 0)
    right_padding = max(window_size - (len(seq) - pos - 1), 0)
    if not padding:
        return seq[pos-window_size+left_padding:pos] + seq[pos:pos+window_size+1-right_padding]
    return [20] * left_padding + seq[pos-window_size+left_padding:pos] + seq[pos:pos+window_size+1-right_padding] + [20] * right_padding

def seq_to_windows(seq, window_size, padding=True):
    """
    Split the protein sequence into a bunch of windows.
    :param seq: Protein sequence where each amino acid is represeented by integer index.
    :param window_size: Number of amino acids to take on each side to form window. 
                        So actual window size will be 2 * window_size + 1.
    :returns: A list of 2 * window_size + 1 sized lists. Part of window that are outside of
                protein sequence are represented by 20.
    """
    return [get_window_at_pos(seq, i, window_size, padding) for i in range(len(seq))]

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


def protein_to_features(seq, ws, local_freq_ws, padding=True):
    num_of_columns = (2 * ws + 1) * 21 #+ 21 + 21 + 1
    non_zero_elem_per_row = (2 * ws + 1) #+ 21 +21 + 1
    data = []
    indices = []
    indptr = [0]
    windows = seq_to_windows(seq, ws, padding)
    local_freqs = seq_to_local_freq(seq, local_freq_ws)
    global_freq = seq_to_freq(seq)
    starting_pos = 0 if padding else ws
    ending_pos = len(seq) if padding else len(seq) - ws
    for i in range(starting_pos, ending_pos):
        '''
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
            indices.append(j)
        '''
        # amino acids in window in one hot representation
        for j in range(len(windows[i])):
            data.append(1)
            indices.append(j * 21 + windows[i][j])
        indptr.append(indptr[-1] +  non_zero_elem_per_row)
    return scsp.csr_matrix((data, indices, indptr), dtype=np.float32, shape = (len(range(starting_pos, ending_pos)), num_of_columns))

if __name__ == '__main__':

    import sys
    if len(sys.argv) < 6:
        print('Usage: python3 preprocessdata.py input_directory output_directory protein_list window_size local_freq_window_size [parse_target]')
        print('Example: python3 preprocessdata.py filtered_data 3484 9 18 X_9_18 y_9_18')
        exit()

    if len(sys.argv) > 5:
        target_available = True
    import time
    print(time.strftime('%Y-%m-%d %H:%M'))

    import os

    protein_list = open(sys.argv[3]).read().split()
    print('Total proteins:', len(protein_list))

    window_size = int(sys.argv[4])
    local_freq_ws = int(sys.argv[5])

    import mxnet as mx

    currently_processing = 0
    for protein in protein_list:
        protein = protein.strip()
    #    print(protein)
        if currently_processing % 100 == 0:
            print('Currently processing: {}'.format(currently_processing))
        currently_processing += 1
        seq = []
        if target_available:
            bfactors = []
        fname = protein.lower()
        for line in open(os.path.join(sys.argv[1], fname)):
            line = line.split()
            a = line[0].strip()
            seq.append(aa_to_index(a))
            if target_available:
                b = float(line[1])
                if(b < 0):
                    print(protein)
                bfactors.append(b)
        X = protein_to_features(seq, window_size, local_freq_ws)
        #bfactors = bfactors[window_size:-window_size]
        assert(X.shape[0] == len(bfactors))
        if target_available:
            bfactors = np.array(bfactors, dtype=np.float32)
            #bfactors = np.log(np.array(bfactors))
            bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
        out_fname_suffix = protein.upper() + '_' + str(window_size) + '_' + str(local_freq_ws)
        # write X
        scsp.save_npz(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), X)
        #np.savez_compressed(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), X=X)
        #mx.nd.save(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), [X])
        # write y
        if target_available:
            np.savez(os.path.join(sys.argv[2], 'y_' + out_fname_suffix), y=bfactors)
            #y = mx.nd.array(bfactors)
            #mx.nd.save(os.path.join(sys.argv[2], 'y_' + out_fname_suffix), [y])
