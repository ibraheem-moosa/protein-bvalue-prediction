import sys
import time
import gc

if len(sys.argv) < 3:
    print('Usage: python3 regression.py X_train y_train X_test y_test [seed]')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
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
X_test = load(sys.argv[3])['X']
print(X_train.shape)
print(X_test.shape)

y_train = load(sys.argv[2])['y']
y_test = load(sys.argv[4])['y']

#y_train = (y_train - y_train.mean()) / y_train.std()
#y_test = (y_test - y_test.mean()) / y_test.std()
print(y_train.shape)
print(y_test.shape)

if len(sys.argv) == 6:
    seed = int(sys.argv[5])
else:
    seed = None

SLIDING_WINDOW_SIZE = (X_test.shape[1]- 1)//43; # must match the equation in rfpreprocessor.py
print("sliding window size:", SLIDING_WINDOW_SIZE)

def test(clf, X_test, y_test):
    y_pred = []
    y_preds = []
    y_trues = []
    ys = []
    y_true = []
    count = 0
    for x, y in zip(X_test, y_test):
        if(x[-1] < -1000):
            count += 1
            if len(ys) > 0:
                y_preds.append(ys)
                y_trues.append(y_true)
            ys = []
            y_true = []
            print("testing sample {}".format(count))
        if(len(ys) >= SLIDING_WINDOW_SIZE):
    	    x[-SLIDING_WINDOW_SIZE:-1] = ys[-SLIDING_WINDOW_SIZE:-1]
        elif(len(ys) > 0):
    	    x[-len(ys)-1:-1] = ys
        y_true.append(y)
        y = clf.predict([x])[0]
        y_pred.append(y)
        ys.append(y)
    
    print("tested {} samples".format(count))
    
    return y_pred, y_preds, y_trues


#clf = RandomForestRegressor(n_estimators=250, n_jobs=4, verbose=5, max_depth=4, max_features='sqrt', random_state=seed)
#clf = LinearRegression(n_jobs=4) #really bad around 0.3 interestingly conc is lower
#clf = GradientBoostingRegressor(verbose=5)
clf = MLPRegressor(hidden_layer_sizes=(1000,), verbose=5)
print(clf)

clf.fit(X_train, y_train)
print("training done.........................")

clf.verbose = 0
clf.n_jobs = 1

y_test_pred, y_test_preds, y_test_trues = test(clf, X_test, y_test) #clf.predict(X_test)
#y_train_pred = test(clf, X_train) #clf.predict(X_train)
print("testing done...................")

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]
def avg_p(y_preds, y_trues):
    pcc = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        pcc += p(y_pred, y_true)
    pcc /= len(y_preds)
    return pcc


#print('Train Pearson Score: {}'.format(p(y_train, y_train_pred)))
#print('Train Error: {}'.format(mean_squared_error(y_train, y_train_pred) / 2.0))
#print('Train R2 Score: {}'.format(clf.score(X_train, y_train)))
#print('Train Data Mean: {} Standard Deviation: {}'.format(np.mean(y_train), np.std(y_train)))
#print('Train Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_pred), np.std(y_train_pred)))

print('Test Pearson Score (avg): {}'.format(avg_p(y_test_trues, y_test_preds)))
print('Test Pearson Score (con): {}'.format(p(y_test, y_test_pred)))
print('Test Error: {}'.format(mean_squared_error(y_test, y_test_pred) / 2.0))
#print('Test R2 Score:: {}'.format(clf.score(X_test, y_test)))
print('Test Data Mean: {} Standard Deviation: {}'.format(np.mean(y_test), np.std(y_test)))
print('Test Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_test_pred), np.std(y_test_pred)))

will_save = input('Do you want to save this classifier? [y/N]')

if 'y' in will_save:
    fname = input('Name of classifier')
    if len(fname) == 0:
        fname = 'classifier-' + time.strftime('%Y-%m-%d %H:%M')
    from sklearn.externals import joblib
    joblib.dump(clf, fname + '.pkl', compress=9)

