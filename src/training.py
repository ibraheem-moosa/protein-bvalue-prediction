import sys
import time
import gc

if len(sys.argv) < 2:
    print('Usage: python3 regression.py X_train y_train')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz')
    exit()
print(time.strftime('%Y-%m-%d %H:%M'))


from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ARDRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.sparse import load_npz, vstack, hstack
from numpy import load
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor


X_train = load(sys.argv[1])['X']
print(X_train.shape)


y_train = load(sys.argv[2])['y']

#y_train = (y_train - y_train.mean()) / y_train.std()
print(y_train.shape)


SLIDING_WINDOW_SIZE = (X_train.shape[1]- 1 - 1 - 21 - 21)//3; # must match the equation in rfpreprocessor.py
print("sliding window size:", SLIDING_WINDOW_SIZE)


clf = RandomForestRegressor(n_estimators=100, n_jobs=4, verbose=0, max_depth=4, max_features='sqrt')
#clf = GradientBoostingRegressor();
print(clf)

clf.fit(X_train, y_train)
gc.collect()
print("training done.........................")

fname = 'classifier-initial' + time.strftime('%Y-%m-%d %H:%M')
from sklearn.externals import joblib
joblib.dump(clf, fname + '.pkl', compress=9)
print("initial classifier saved as ", fname)

