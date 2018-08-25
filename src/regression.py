import sys
import time
import gc
from sklearn.linear_model import LinearRegression, RidgeCV, BayesianRidge, ARDRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.sparse import load_npz, vstack, hstack
from numpy import load
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr



def plot_true_and_prediction(y_true, y_pred):
    fig = plt.figure()
    plt.title('True Green, Predicted Yellow')
    plt.plot(y_pred, 'y-')
    plt.plot(y_true, 'g-')
    plt.show()

def test(clf, X_test, y_test, plot_reversed=False):
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
                if random.random() < 0.00:
                    if plot_reversed:
                        plot_true_and_prediction(list(reversed(y_true)), list(reversed(ys)))
                    else:
                        plot_true_and_prediction(y_true, ys)
            ys = []
            y_true = []
            if count % 100 == 0:
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

def ndarrays_from_left_right(y_lefts, y_rights, y_trues, ws=1):
    total_length = sum(map(len, y_trues))
    X = np.zeros((total_length, 2))
    y = np.zeros((total_length,))
    current_pos = 0
    for y_l, y_r in zip(y_lefts, y_rights):
        for l, r in zip(y_l, y_r):
            X[current_pos] = [l, r]
            current_pos += 1
    current_pos = 0
    for y_t in y_trues:
        y[current_pos:current_pos+len(y_t)] = y_t
        current_pos += len(y_t)
    return X, y


if len(sys.argv) < 3:
    print('Usage: python3 regression.py X_train y_train X_test y_test [seed]')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
    exit()
print(time.strftime('%Y-%m-%d %H:%M'))


X_train_left = load(sys.argv[1] + '_left.npz')['X']
X_train_right = load(sys.argv[1] + '_right.npz')['X']
X_test_left = load(sys.argv[3] + '_left.npz')['X']
X_test_right = load(sys.argv[3] + '_right.npz')['X']
print(X_train_left.shape)
print(X_train_right.shape)
print(X_test_left.shape)
print(X_test_right.shape)

y_train_left = load(sys.argv[2] + '_left.npz')['y']
y_train_right = load(sys.argv[2] + '_right.npz')['y']
y_test_left = load(sys.argv[4] + '_left.npz')['y']
y_test_right = load(sys.argv[4] + '_right.npz')['y']

y_test = y_test_left

#y_train = (y_train - y_train.mean()) / y_train.std()
#y_test = (y_test - y_test.mean()) / y_test.std()
print(y_train_left.shape)
print(y_train_right.shape)
print(y_test_left.shape)
print(y_test_right.shape)

if len(sys.argv) == 6:
    seed = int(sys.argv[5])
else:
    seed = None

SLIDING_WINDOW_SIZE = (X_test_left.shape[1]- 1)//43; # must match the equation in rfpreprocessor.py
#SLIDING_WINDOW_SIZE = (X_test_left.shape[1]- 1)//3; # must match the equation in rfpreprocessor.py
print("sliding window size:", SLIDING_WINDOW_SIZE)


#clf = RandomForestRegressor(n_estimators=250, n_jobs=4, verbose=5, max_depth=4, max_features='sqrt', random_state=seed)
#clf = LinearRegression(n_jobs=4) #really bad around 0.3 interestingly conc is lower
#clf_left = GradientBoostingRegressor(n_estimators=1500, verbose=5)
#clf_right = GradientBoostingRegressor(n_estimators=1500, verbose=5)
clf_left = MLPRegressor(hidden_layer_sizes=(10,), verbose=5)
clf_right = MLPRegressor(hidden_layer_sizes=(10,), verbose=5)
print(clf_left)

clf_left.fit(X_train_left, y_train_left)
clf_right.fit(X_train_right, y_train_right)
print("training done.........................")

clf_left.verbose = 0
clf_right.verbose = 0
clf_left.n_jobs = 1
clf_right.n_jobs = 1

y_train_left_pred, y_train_left_preds, y_train_left_trues = test(clf_left, X_train_left, y_train_left) #clf.predict(X_test)
y_train_right_pred, y_train_right_preds, y_train_right_trues = test(clf_right, X_train_right, y_train_right, plot_reversed=True)

X, y = ndarrays_from_left_right(y_train_left_preds, y_train_right_preds, y_train_left_trues)
#clf = LinearRegression(n_jobs=4) 
#clf = RidgeCV()
clf = MLPRegressor(hidden_layer_sizes=(4,), verbose=5)
clf.fit(X, y)
print(clf.coefs_)
clf.verbose=0
print("second stage training done.........................")

y_test_left_pred, y_test_left_preds, y_test_left_trues = test(clf_left, X_test_left, y_test_left) #clf.predict(X_test)
y_test_right_pred, y_test_right_preds, y_test_right_trues = test(clf_right, X_test_right, y_test_right, plot_reversed=True) #clf.predict(X_test)

y_test_preds_clf = []
y_test_preds_avg = []
for y_l, y_r in zip(y_test_left_preds, y_test_right_preds):
    y_test_preds_clf.append(clf.predict(np.stack((np.array(y_l), np.array(list(reversed(y_r)))), axis=-1)))
    y_test_preds_avg.append(0.5 * (np.array(y_l) + np.array(list(reversed(y_r)))))

y_test_trues = []
for y_l, y_r in zip(y_test_left_trues, y_test_right_trues):
    y_test_trues.append(0.5 * (np.array(y_l) + np.array(list(reversed(y_r)))))


print("testing done...................")

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]
def avg_p(y_preds, y_trues):
    pcc = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        pcc += p(y_pred, y_true)
    pcc /= len(y_preds)
    return pcc

def avg_mse(y_preds, y_trues):
    mse = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        mse += mean_squared_error(y_pred, y_true)
    mse /= len(y_preds)
    return mse



#print('Train Pearson Score: {}'.format(p(y_train, y_train_pred)))
#print('Train Error: {}'.format(mean_squared_error(y_train, y_train_pred) / 2.0))
#print('Train R2 Score: {}'.format(clf.score(X_train, y_train)))
#print('Train Data Mean: {} Standard Deviation: {}'.format(np.mean(y_train), np.std(y_train)))
#print('Train Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_pred), np.std(y_train_pred)))

print('Test Left Pearson Score (avg): {}'.format(avg_p(y_test_left_trues, y_test_left_preds)))
print('Test Left Error: {}'.format(mean_squared_error(y_test_left, y_test_left_pred) / 2.0))

print('Test Right Pearson Score (avg): {}'.format(avg_p(y_test_right_trues, y_test_right_preds)))
print('Test Right Error: {}'.format(mean_squared_error(y_test_right, y_test_right_pred) / 2.0))

print('Test Classifier Combined Pearson Score (avg): {}'.format(avg_p(y_test_trues, y_test_preds_clf)))
print('Test Classifier Combined Error: {}'.format(avg_mse(y_test_trues, y_test_preds_clf) / 2.0))

print('Test Average Combined Pearson Score (avg): {}'.format(avg_p(y_test_trues, y_test_preds_avg)))
print('Test Average Combined Error: {}'.format(avg_mse(y_test_trues, y_test_preds_avg) / 2.0))


#print('Test R2 Score:: {}'.format(clf.score(X_test, y_test)))
print('Test Data Mean: {} Standard Deviation: {}'.format(np.mean(y_test_left), np.std(y_test_left)))
print('Test Left Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_test_left_pred), np.std(y_test_left_pred)))
print('Test Right Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_test_right_pred), np.std(y_test_right_pred)))


will_save = input('Do you want to save this classifier? [y/N]')

if 'y' in will_save:
    fname = input('Name of classifier')
    if len(fname) == 0:
        fname = 'classifier-' + time.strftime('%Y-%m-%d %H:%M')
    from sklearn.externals import joblib
    joblib.dump(clf, fname + '.pkl', compress=9)

