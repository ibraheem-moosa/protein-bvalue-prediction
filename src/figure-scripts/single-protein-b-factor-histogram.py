import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import scipy
import random
# sns.set()
if(len(sys.argv) < 2):
    print("insufficient input")
    exit(0)

random.seed(42)

files = os.listdir(sys.argv[1])

random.shuffle(files)


for f in files:

    data = np.genfromtxt(os.path.join(sys.argv[1], f))
    bval = data[:,1]
    mean = bval.mean()
    stdev = bval.std()
    
    bval = (bval - mean) / stdev

    dist = pd.DataFrame(bval)
    # dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)

    print("here...1")


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
    sns.set_color_codes("bright")
    sns.distplot(bval, fit=scipy.stats.norm, kde=False, color='#1F77B4')

    print(f, mean, stdev)
    plt.show()
    btnpress = plt.waitforbuttonpress(0.1)
    if btnpress:
        plt.waitforbuttonpress(-1)