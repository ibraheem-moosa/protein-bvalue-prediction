import time
print(time.strftime('%Y-%m-%d %H:%M'))

import sys
import numpy as np

if len(sys.argv) < 5:
    print('Usage: python3 regression.py num_of_data input.txt input_dim target.txt')
    exit()

X = np.empty((int(sys.argv[1]), int(sys.argv[3])), dtype=np.float32)
print(X.shape)
y = np.empty((X.shape[0],), dtype=np.float32)
print(y.shape)


with open(sys.argv[2]) as f:
    for i, line in enumerate(f):
        X[i] = np.array(list(map(float, line.split())))


with open(sys.argv[4]) as f:
    for i, line in enumerate(f):
        y[i] = (float(line))

y = (y - np.mean(y)) / np.std(y)
# data input and preprocessing done
'''
from sklearn.dummy import DummyRegressor
rom sklearn.ensemble import BaggingRegressor
'''
from sklearn.neural_network import MLPRegressor

#dummy_clf = DummyRegressor(strategy='mean')

clf = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', 
                    alpha=0.1, batch_size=256, early_stopping=True, 
                    learning_rate_init=0.01, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=200, tol=1e-8, verbose=True, validation_fraction=0.1)

print(clf.hidden_layer_sizes)
#clf = BaggingRegressor(clf, n_estimators=10, 
#                       max_samples=0.75, verbose=5, n_jobs=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

import gc
gc.collect()

clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

gc.collect()

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]

print(p(y_train, y_train_pred))
#print(mean_squared_error(y_train, y_train_pred))
print(p(y_test, y_test_pred))
#print(mean_squared_error(y_test, y_test_pred))

'''
# grid search

from sklearn.metrics import make_scorer
pcc = make_scorer(p)
mse = make_scorer(mean_squared_error, greater_is_better=False)

from sklearn.model_selection import GridSearchCV
p_grid = {'learning_rate_init': [0.001],
          'alpha': [0.1, 0.01],
          'batch_size': [256, 512],
          'hidden_layer_sizes': [(8,), (16,), (32,), (64,)]}

gscv = GridSearchCV(clf, param_grid=p_grid, scoring={'Pearson':pcc,'MSE':mse}, 
                        verbose=5, cv=3, refit='Pearson', n_jobs=1)


gscv.fit(X_train,y_train)
print(gscv.best_params_)
print(gscv.best_score_)
print(gscv.score(X_test, y_test))
import pandas as pd

with open('grid_search_one_layer_ws_twelve.csv', 'a') as f:
    pd.DataFrame(gscv.cv_results_).to_csv(f, header=True)

print('Grid Search CV Done')
'''
#from sklearn.model_selection import cross_val_score
#mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
#score = cross_val_score(clf, X, y, cv=5, verbose=1, scoring=mse_scorer)

#from math import sqrt
#print(score)
#print(sqrt(-sum(score)/len(score)))
#print(sum(y)/len(y))


#clf.fit(X,y)
#print(clf.n_features_)
#print(clf.feature_importances_)
#important_features = [i for i in range(clf.n_features_) if clf.feature_importances_[i] > 1e-2]
#print(important_features)

