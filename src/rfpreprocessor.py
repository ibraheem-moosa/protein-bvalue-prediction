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

def get_window_at_pos(seq, pos, window_size, padding_value=20):
    left_padding = max((window_size - pos), 0)
    right_padding = max(window_size - (len(seq) - pos - 1), 0)
    return [padding_value] * left_padding + seq[pos-window_size+left_padding:pos] + seq[pos:pos+window_size+1-right_padding] + [padding_value] * right_padding

def seq_to_windows(seq, window_size, padding_value=20):
    """
    Split the protein sequence into a bunch of windows.
    :param seq: Protein sequence where each amino acid is represeented by integer index.
    :param window_size: Number of amino acids to take on each side to form window. 
                        So actual window size will be 2 * window_size + 1.
    :returns: A list of 2 * window_size + 1 sized lists. Part of window that are outside of
                protein sequence are represented by 20.
    """
    return [get_window_at_pos(seq, i, window_size, padding_value) for i in range(len(seq))]

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


def seq_to_markov_probability(seq, freq):
    prob = [[0.0] * 21]*21
    prev = -1
    for aa in seq:
        prob[prev][aa] += 1
        prev = aa
    for i in range(21):
    	for j in range(21):
    	    if(freq[i] != 0):
    	        prob[i][j]= prob[i][j] / freq[i]
    return prob


def get_b_factor_window_at_position(seq, pos, window_size):
    left_padding = max((window_size - pos), 0)
    #right_padding = max(window_size - (len(seq) - pos - 1), 0)
    return [-10000] * left_padding + seq[pos-window_size+left_padding:pos] #+ seq[pos:pos+window_size+1-right_padding] + [20] * right_padding


def get_b_factor_window_seq(seq, window_size):
    """
    Split the b factor sequence into a bunch of windows.
    :param seq: b factor sequence of real value
    :param window_size: Number of amino acids to take on each side to form window. 
                        So actual window size will be 2 * window_size + 1.
    :returns: A list of 2 * window_size + 1 sized lists. Part of window that are outside of
                protein sequence are represented by 20.
    """
    return [get_b_factor_window_at_position(seq, i, window_size) for i in range(len(seq))]


def protein_to_features(seq, ws, local_freq_ws, bfactors):
    #ohtonum
    #num_of_columns = (2 * ws + 1) + ws #+21*21 
    num_of_columns = (2 * ws + 1) * 21 + ws #+21*21 
    non_zero_elem_per_row = (2 * ws + 1) + ws
    data = []
    indices = []
    indptr = [0]
    windows = seq_to_windows(seq, ws)
    local_freqs = seq_to_local_freq(seq, local_freq_ws)
    global_freq = seq_to_freq(seq)
    #global_markov_probability = seq_to_markov_probability(seq, global_freq)
    bfactor_windows = get_b_factor_window_seq(bfactors, ws)
    
    for i in range(len(seq)):
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
            indices.append(1 + 21 + j)
        '''
        # amino acids in window in numerical representation
        for j in range(len(windows[i])):
            #ohtonum
            #data.append(windows[i][j])
            data.append(1)
            #ohtonum
            #indices.append(j)
            indices.append(j * 21 + windows[i][j])
        
        for j in range(len(bfactor_windows[i])):
            data.append(bfactor_windows[i][j])
            #ohtonum
            #indices.append(len(windows[i]) + j) 
            indices.append(21* len(windows[i]) + j) 
            
        
        '''
        # markov probability
        for j in range(21):
            for k in range(21):
                data.append(global_markov_probability[j][k])
                indices.append(1 + 21 + 21 + len(windows[i]) + j*21 + k)
        '''

        indptr.append(indptr[-1] +  non_zero_elem_per_row)
    #ohtonum    
    return scsp.csr_matrix((data, indices, indptr), dtype=np.float32, shape = (len(seq), num_of_columns)).todense()
    #return np.array(data).reshape(-1,num_of_columns)



def ndarray_from_files(files, window_size, local_freq_ws, reverse=False):
    y = []
    currently_processing = 0
    #ohtonum
    #num_of_columns = (2 * window_size + 1) + window_size #+ 21*21
    num_of_columns = (2 * window_size + 1) * 21 + window_size #+ 21*21
    non_zero_elem_per_row = num_of_columns

    X = np.zeros((0, num_of_columns))
    print("empty x shape", X.shape)
    for f in files:
        if currently_processing % 100 == 0:
            print('Currently processing: {}'.format(currently_processing))
        currently_processing += 1
        seq = []
        bfactors = []
        for line in open(os.path.join(sys.argv[1], f)):
            line = line.split()
            a = line[0].strip()
            seq.append(aa_to_index(a))
            b = float(line[1])
            bfactors.append(b)
        if reverse:
            seq.reverse()
            bfactors.reverse()
        X = np.vstack((X, protein_to_features(seq, window_size, local_freq_ws, bfactors)))
    
        #bfactors = np.log(np.array(bfactors))
        #bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
        y.extend(bfactors)
    y = np.array(y, dtype=np.float32)
    return X, y

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 6:
        print('Usage: python3 preprocessdata.py input_directory num_of_proteins window_size local_freq_window_size X y')
        print('Example: python3 preprocessdata.py filtered_data 3484 9 18 X_9_18 y_9_18')
        exit()

    import time
    print(time.strftime('%Y-%m-%d %H:%M'))

    import os

    num_of_proteins = int(sys.argv[2])
    print('Total files: {}'.format(len(os.listdir(sys.argv[1]))))
    files = os.listdir(sys.argv[1])
    random.seed(42)
    random.shuffle(files)
    if num_of_proteins < len(files):
        files = random.sample(files, num_of_proteins)

    num_of_train = int(0.9 * len(files))
    train_files = files[:num_of_train]
    test_files = files[num_of_train:]


    window_size = int(sys.argv[3])
    local_freq_ws = int(sys.argv[4])

    X_train_left, y_train_left = ndarray_from_files(train_files, window_size, local_freq_ws, reverse=False)
    X_train_right, y_train_right = ndarray_from_files(train_files, window_size, local_freq_ws, reverse=True)
    X_test_left, y_test_left =  ndarray_from_files(test_files, window_size, local_freq_ws, reverse=False)
    X_test_right, y_test_right =  ndarray_from_files(test_files, window_size, local_freq_ws, reverse=True)

    # write X
    np.savez_compressed(sys.argv[5] + '_train_left', X=X_train_left)
    np.savez_compressed(sys.argv[5] + '_train_right', X=X_train_right)
    np.savez_compressed(sys.argv[5] + '_test_left', X=X_test_left)
    np.savez_compressed(sys.argv[5] + '_test_right', X=X_test_right)
    # write y
    np.savez_compressed(sys.argv[6] + '_train_left', y=y_train_left)
    np.savez_compressed(sys.argv[6] + '_train_right', y=y_train_right)
    np.savez_compressed(sys.argv[6] + '_test_left', y=y_test_left)
    np.savez_compressed(sys.argv[6] + '_test_right', y=y_test_right)

