import sys
import time
import gc

if len(sys.argv) < 3:
    print('Usage: python3 test_using_saved_clf.py clf_file X_val y_val')
    print('Example: python3 regression.py clf.pkl X_val_9_18.npz y_val_9_18.npz')
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
from sklearn.externals import joblib

classifier_file = sys.argv[1]
X_test = load(sys.argv[2])['X']
print(X_test.shape)

y_test = load(sys.argv[3])['y']
#y_test = (y_test - y_test.mean()) / y_test.std()
print(y_test.shape)

SLIDING_WINDOW_SIZE = (X_test.shape[1]- 1 - 1 - 21 - 21)//3; # must match the equation in rfpreprocessor.py
print("sliding window size:", SLIDING_WINDOW_SIZE)


'''
validate() returns modified data points that will be used to train the final clf and predicted y values
'''

def validate(clf, X_test):
    y_pred = []
    ys = []
    count = 0
    for i in range(len(X_test)):
        if(X_test[i][-1] < -1000):
            count += 1
            ys = []
            print("testing sample {}".format(count))
        if(len(ys) >= SLIDING_WINDOW_SIZE):
    	    X_test[i][-SLIDING_WINDOW_SIZE:] = ys[-SLIDING_WINDOW_SIZE:] # replacing with predicted value
        elif(len(ys) > 0):
    	    X_test[i][-len(ys):] = ys
        y = clf.predict([X_test[i]])
        y_pred.append(y[0])
        ys.append(y[0])
    
    print("tested {} samples".format(count))
    
    return y_pred, X_test


clf = joblib.load(classifier_file)

clf.n_jobs = 1

print("classifier loaded.........................")

y_test_pred, new_X_train = validate(clf, X_test) #clf.predict(X_test)
#y_train_pred = test(clf, X_train) #clf.predict(X_train)
gc.collect()

length = len(y_test)

print("testing done...................")




from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]



#print('Train Pearson Score: {}'.format(p(y_train, y_train_pred)))
#print('Train Error: {}'.format(mean_squared_error(y_train, y_train_pred) / 2.0))
#print('Train R2 Score: {}'.format(clf.score(X_train, y_train)))
#print('Train Data Mean: {} Standard Deviation: {}'.format(np.mean(y_train), np.std(y_train)))
#print('Train Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_pred), np.std(y_train_pred)))

print('Val Pearson Score: {}'.format(p(y_test, y_test_pred)))
print('Val Error: {}'.format(mean_squared_error(y_test, y_test_pred) / 2.0))
#print('Test R2 Score:: {}'.format(clf.score(X_test, y_test)))
print('Val Data Mean: {} Standard Deviation: {}'.format(np.mean(y_test), np.std(y_test)))
print('Val Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_test_pred), np.std(y_test_pred)))









clf = RandomForestRegressor(n_estimators=100, n_jobs=4, verbose=0, max_depth=4, max_features='sqrt')
#clf = GradientBoostingRegressor();
print(clf)

clf.fit(new_X_train, y_test)
gc.collect()

print("new classifier trained")





fname = 'classifier-final' + time.strftime('%Y-%m-%d %H:%M')
from sklearn.externals import joblib
joblib.dump(clf, fname + '.pkl', compress=9)
print("final classifier saved as ", fname)




print(time.strftime('%Y-%m-%d %H:%M'))
