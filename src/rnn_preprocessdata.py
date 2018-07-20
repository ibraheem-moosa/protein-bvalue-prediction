from Bio.PDB import Polypeptide
import numpy as np
import random
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

def protein_to_features(seq):
    num_of_columns = 21
    non_zero_elem_per_row = 1
    data = []
    indices = []
    indptr = [0]
    for i in range(len(seq)):
        data.append(1)
        indices.append(seq[i])
        indptr.append(indptr[-1] +  non_zero_elem_per_row)
    return scsp.csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(seq), num_of_columns))

if __name__ == '__main__':

    import sys
    if len(sys.argv) < 4:
        print('Usage: python3 preprocessdata.py input_directory output_directory protein_list [parse_target]')
        print('Example: python3 preprocessdata.py filtered_data 3484 9 18 X_9_18 y_9_18')
        exit()

    if len(sys.argv) == 5:
        target_available = True
    else:
        target_available = False
    import time
    print(time.strftime('%Y-%m-%d %H:%M'))

    import os

    protein_list = open(sys.argv[3]).read().split()
    print('Total proteins:', len(protein_list))

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
                if(b <= 0):
                    print(protein)
                bfactors.append(b)
        X = protein_to_features(seq)
        assert(X.shape[0] == len(bfactors))
        if target_available:
            bfactors = np.log(np.array(bfactors))
            bfactors = (bfactors - np.mean(bfactors)) / np.std(bfactors)
            bfactors = np.array(bfactors, dtype=np.float32)

        out_fname_suffix = protein.upper() + '_rnn_'
        # write X
        scsp.save_npz(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), X)
        #np.savez_compressed(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), X=X)
        #mx.nd.save(os.path.join(sys.argv[2], 'X_' + out_fname_suffix), [X])
        # write y
        if target_available:
            np.savez_compressed(os.path.join(sys.argv[2], 'y_' + out_fname_suffix), y=bfactors)
            #y = mx.nd.array(bfactors)
            #mx.nd.save(os.path.join(sys.argv[2], 'y_' + out_fname_suffix), [y])
