import os
import sys
from Bio.PDB import *

if len(sys.argv) < 2:
    print('Usage: python3 filter.py input_directory output_directory')
    print('Example: python3 filter.py data filtered_data')
    exit()

p = MMCIFParser()

count = 0
for f in os.listdir(sys.argv[1]):
    if count % 100 == 0:
        print('Count:',count)
    count += 1
    print('Processing: ' + f)
    s = p.get_structure('X', os.path.join(sys.argv[1], f))
    for m in s:
        num_of_chains = len([c for c in m])
        if num_of_chains == 1:
            print('Skipping')
            continue
        chain_num = 0
        for c in m:
            o = open(os.path.join(sys.argv[2], f[:-4] + '-' + str(chain_num)), 'w')
            chain_num += 1
            for r in c:
                #print(r.get_id())
                if Polypeptide.is_aa(r.get_resname()) and 'CA' in r:
                    #print(r.get_resname())
                    o.write(r.get_resname() + ' ' + str(r['CA'].get_bfactor()) + '\n')
                #else:
                #    print('Skipped' + r.get_resname())
            o.close()

