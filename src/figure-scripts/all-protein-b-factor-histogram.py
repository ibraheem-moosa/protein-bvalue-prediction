import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import scipy
import random

if(len(sys.argv) < 2):
    print("insufficient input")
    exit(0)

random.seed(42)

files = os.listdir(sys.argv[1])

random.shuffle(files)

bval = []
for f in files:
    data = np.genfromtxt(os.path.join(sys.argv[1], f))
    b = data[:,1]
    mean = b.mean()
    stdev = b.std()
    b = (b - mean) / stdev

    bval.extend(b)

    # dist = pd.DataFrame(bval)
    # dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)



fig, ax = plt.subplots()

print("here...2")
# dist.plot.kde(ax=ax, legend=False, title='Histogram: A vs. B')


print("here...3")
# dist.plot.hist(density=True, ax=ax, bins = 20)

print("here...4")
ax.set_ylabel('Probability')
ax.set_xlabel('B-factor values')


print("here...5")
ax.grid(axis='y')


print("here...6")
# ax.set_facecolor('#d8dcd6')


# plt.show()
# print("done")


sns.distplot(bval, fit=scipy.stats.norm, kde=False)

print(f, mean, stdev)
plt.show()
