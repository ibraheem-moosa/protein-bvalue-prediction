import sys

f = open(sys.argv[1])

l = [l for l in f]

l = sorted(l, key = lambda s:float(s.split()[2]))

for lx in l:
    print(lx)
