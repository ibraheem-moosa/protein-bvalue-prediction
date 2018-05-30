from Bio.PDB import *
import sys
import os

if (len(sys.argv) < 2):
    print('Usage: python3 download.py pdb_ids_file output_directory')
    exit()
pdb_ids = open(sys.argv[1])
pdb_ids = ''.join([line for line in pdb_ids])
pdb_ids = pdb_ids.split(',')
pdb_ids = list(map(lambda s:s.strip(), pdb_ids))

pdbl = PDBList()

pdbl.download_pdb_files(pdb_ids, pdir=sys.argv[2])
