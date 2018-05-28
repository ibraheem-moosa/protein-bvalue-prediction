import os
import sys
from Bio.PDB import *

if len(sys.argv) < 2:
    print('Usage: python3 filter.py input_directory output_directory')
    exit()

p = MMCIFParser()

for f in os.listdir(sys.argv[1]):
    print('Processing: ' + f)
    s = p.get_structure('X', os.path.join(sys.argv[1], f))
    o = open(os.path.join(sys.argv[2], f[:-4]), 'w')
    for m in s:
        num_of_chains = len([c for c in m])
        if num_of_chains > 1:
            print('Skipping')
            continue
        for c in m:
            for r in c:
                #print(r.get_id())
                if Polypeptide.is_aa(r.get_resname()) and 'CA' in r:
                    #print(r.get_resname())
                    o.write(r.get_resname() + ' ' + str(r['CA'].get_bfactor()) + '\n')
                #else:
                #    print('Skipped' + r.get_resname())
    o.close()
