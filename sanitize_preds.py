import os
import sys

files = os.listdir(sys.argv[1])
files = [os.path.join(sys.argv[1], f) for f in files]

for fname in files:
    l = []
    with open(fname) as f:


        lines = [float(s.split('[')[1].split(']')[0]) for s in f]
        l.extend(lines)
        
    with open(fname, "w") as f:
        for line in l:
            f.write('{}\n'.format(line))

