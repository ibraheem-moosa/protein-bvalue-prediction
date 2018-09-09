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
from rfpreprocessor import seq_to_windows
from sklearn.externals import joblib

def plot_true_and_prediction(y_true, y_pred):
    fig = plt.figure()
    plt.title('True Green, Predicted Yellow')
    plt.plot(y_pred, 'y-')
    plt.plot(y_true, 'g-')
    plt.show()

def test(clf, X_test, y_test, plot_reversed=False, isTest=False):
    y_pred = []
    y_preds = []
    y_trues = []
    ys = []
    y_true = []
    count = 0
    
    newX = []
    for x, y in zip(X_test, y_test):
        if(x[-1] < -1000):
            count += 1
            if len(ys) > 0:
                y_preds.append(ys)
                y_trues.append(y_true)
                if random.random() < 0.01 and isTest:
                    if plot_reversed:
                        plot_true_and_prediction(list(reversed(y_true)), list(reversed(ys)))
                    else:
                        plot_true_and_prediction(y_true, ys)
            ys = []
            y_true = []
            #if count % 100 == 0:
            #    print("testing sample {}".format(count))
        if(len(ys) >= SLIDING_WINDOW_SIZE):
    	    x[-SLIDING_WINDOW_SIZE:] = ys[-SLIDING_WINDOW_SIZE:]
        elif(len(ys) > 0):
    	    x[-len(ys):] = ys
        y_true.append(y)
        y = clf.predict([x])[0]
        y_pred.append(y)
        ys.append(y)
        newX.append(x)
    
    print("tested {} samples".format(count))
    
    return y_pred, y_preds, y_trues, newX

def ndarrays_from_left_right(y_lefts, y_rights, y_trues, ws=0):
    total_length = sum(map(len, y_trues))
    X = np.zeros((total_length, 2 * (2 * ws + 1)))
    y = np.zeros((total_length,))
    current_pos = 0
    for y_l, y_r in zip(y_lefts, y_rights):
        y_r = list(reversed(y_r))
        y_l = seq_to_windows(y_l, ws, padding_value=-10000)
        y_r = seq_to_windows(y_r, ws, padding_value=-10000)
        for l, r in zip(y_l, y_r):
            X[current_pos] = l + r
            current_pos += 1
    current_pos = 0
    for y_t in y_trues:
        y[current_pos:current_pos+len(y_t)] = y_t
        current_pos += len(y_t)
    return X, y


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


def final_test(clf, y_test_left_preds, y_test_right_preds):
    y_test_preds_clf = []
    y_test_preds_avg = []
    for y_l, y_r in zip(y_test_left_preds, y_test_right_preds):
        y_r = list(reversed(y_r))
        y_test_preds_avg.append(0.5 * (np.array(y_l) + np.array(y_r)))
        y_l = seq_to_windows(y_l, SLIDING_WINDOW_SIZE, padding_value=-10000)
        y_r = seq_to_windows(y_r, SLIDING_WINDOW_SIZE, padding_value=-10000)
        clf_pred = []
        for l, r in zip(y_l, y_r):
            clf_pred.append(clf.predict([l + r])[0])
        y_test_preds_clf.append(clf_pred)
    return y_test_preds_clf, y_test_preds_avg


DIR = 'clfs_bidirectional/'

if len(sys.argv) < 4:
    print('Usage: python3 validation_bidirectional.py X y  clf_to_save [clf_to_load] [seed]')
    print('Example: python3 validation_bidirectional.py X.npz y.npz clf_save [clf_load] [10]')
    exit()
print(time.strftime('%Y-%m-%d %H:%M'))


X_train_left = load(sys.argv[1] + '_train_left.npz')['X']
X_train_right = load(sys.argv[1] + '_train_right.npz')['X']

X_test_left = load(sys.argv[1] + '_test_left.npz')['X']
X_test_right = load(sys.argv[1] + '_test_right.npz')['X']

print(X_train_left.shape)
print(X_train_right.shape)

y_train_left = load(sys.argv[2] + '_train_left.npz')['y']
y_train_right = load(sys.argv[2] + '_train_right.npz')['y']

y_test_left = load(sys.argv[2] + '_test_left.npz')['y']
y_test_right = load(sys.argv[2] + '_test_right.npz')['y']





y_test = y_test_left


print(y_train_left.shape)
print(y_train_right.shape)
print(y_test_left.shape)
print(y_test_right.shape)

if(len(sys.argv) > 4):
    classifier_file = DIR+sys.argv[4]
    clf_left = joblib.load(classifier_file+'_left.pkl')
    clf_right = joblib.load(classifier_file+'_right.pkl')
    clf = joblib.load(classifier_file+'.pkl')

    clf_left.max_features = 'sqrt'
    clf_left.warm_start = False
    clf_left.n_estimators = 100
    clf.n_estimators = 100
    clf.max_features = 'sqrt'
else:
    clf_left = GradientBoostingRegressor(n_estimators= 10, warm_start = True)
    clf_right = GradientBoostingRegressor(n_estimators= 10, warm_start = True)
    clf_left.fit(X_train_left, y_train_left)
    clf_right.fit(X_train_right, y_train_right)
    clf = GradientBoostingRegressor(n_estimators= 2)
    print("training with new clf done.........................")

if len(sys.argv) > 5:
    seed = int(sys.argv[5])
else:
    seed = None

#ohtonum
#SLIDING_WINDOW_SIZE = (X_test_left.shape[1]- 1)//43; # must match the equation in rfpreprocessor.py
SLIDING_WINDOW_SIZE = (X_test_left.shape[1]- 1-1-21-21)//3; # must match the equation in rfpreprocessor.py
print("sliding window size:", SLIDING_WINDOW_SIZE)


# clf_left = GradientBoostingRegressor(n_estimators=250, verbose=5, random_state=seed, max_depth=3, learning_rate=0.1)
# clf_right = GradientBoostingRegressor(n_estimators=250, verbose=5, random_state=seed, max_depth=3, learning_rate=0.1)
#clf_left = MLPRegressor(hidden_layer_sizes=(100,), verbose=5)
#clf_right = MLPRegressor(hidden_layer_sizes=(100,), verbose=5)
print(clf_left)


# clf_left.verbose = 0
# clf_right.verbose = 0
# clf_left.n_jobs = 1
# clf_right.n_jobs = 1

y_train_left_pred, y_train_left_preds, y_train_left_trues, newX_left = test(clf_left, X_train_left, y_train_left) #clf.predict(X_test)
y_train_right_pred, y_train_right_preds, y_train_right_trues, newX_right = test(clf_right, X_train_right, y_train_right, plot_reversed=True)

y_train_trues = y_train_left



if(len(sys.argv) > 4):
	y_train_preds_clf, y_train_preds_avg = final_test(clf, y_train_left_preds, y_train_right_preds)

	print('train Left Pearson Score (avg): {}'.format(avg_p(y_train_left_trues, y_train_left_preds)))
	print('train Left Error: {}'.format(mean_squared_error(y_train_left, y_train_left_pred) / 2.0))

	print('train Right Pearson Score (avg): {}'.format(avg_p(y_train_right_trues, y_train_right_preds)))
	print('train Right Error: {}'.format(mean_squared_error(y_train_right, y_train_right_pred) / 2.0))

	# print('train Classifier Combined Pearson Score (avg): {}'.format(avg_p(y_train_trues, y_train_preds_clf)))
	# print('train Classifier Combined Error: {}'.format(avg_mse(y_train_trues, y_train_preds_clf) / 2.0))

	# print('train Average Combined Pearson Score (avg): {}'.format(avg_p(y_train_trues, y_train_preds_avg)))
	# print('train Average Combined Error: {}'.format(avg_mse(y_train_trues, y_train_preds_avg) / 2.0))


	#print('train R2 Score:: {}'.format(clf.score(X_train, y_train)))
	print('train Data Mean: {} Standard Deviation: {}'.format(np.mean(y_train_left), np.std(y_train_left)))
	print('train Left Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_left_pred), np.std(y_train_left_pred)))
	print('train Right Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_right_pred), np.std(y_train_right_pred)))


gc.collect()



clf_left.n_estimators += 1
clf_right.n_estimators += 1

clf_left.fit(newX_left, y_train_left)
clf_right.fit(newX_right, y_train_right)



X, y = ndarrays_from_left_right(y_train_left_preds, y_train_right_preds, y_train_left_trues, ws=SLIDING_WINDOW_SIZE)


gc.collect()
#clf = LinearRegression(n_jobs=4) 
#clf = RidgeCV()
# clf = GradientBoostingRegressor(verbose=5, n_estimators=500, random_state=seed, learning_rate=0.2)
clf.fit(X, y)
clf.verbose=0
print("second stage training done.........................**************")

y_test_left_pred, y_test_left_preds, y_test_left_trues, dummy = test(clf_left, X_test_left, y_test_left, isTest=True) #clf.predict(X_test)
y_test_right_pred, y_test_right_preds, y_test_right_trues, dummy = test(clf_right, X_test_right, y_test_right, plot_reversed=True, isTest=True) #clf.predict(X_test)

y_test_preds_clf, y_test_preds_avg = final_test(clf, y_test_left_preds, y_test_right_preds)



y_test_trues = y_test_left_trues
# for y_l, y_r in zip(y_test_left_trues, y_test_right_trues):
#     y_test_trues.append(0.5 * (np.array(y_l) + np.array(list(reversed(y_r)))))


gc.collect()

print("testing done...................")




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



joblib.dump(clf, DIR+sys.argv[3] + '.pkl', compress=9)
joblib.dump(clf_left, DIR+sys.argv[3] + '_left.pkl', compress=9)
joblib.dump(clf_right, DIR+sys.argv[3] + '_right.pkl', compress=9)

print(time.strftime('%Y-%m-%d %H:%M'))

