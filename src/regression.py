import time
print(time.strftime('%Y-%m-%d %H:%M'))

import sys
import numpy as np

if len(sys.argv) < 3:
    print('Usage: python3 regression.py X y [seed]')
    print('Example: python3 regression.py X_9_18.npz y_9_18.npz 10')
    exit()
from scipy.sparse import load_npz
X = load_npz(sys.argv[1])
print(X.shape)
from numpy import load
y = load(sys.argv[2])
y = y['y']
print(y.shape)
if len(sys.argv) == 4:
    seed = int(sys.argv[3])
else:
    seed = None
# data input and preprocessing done
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(hidden_layer_sizes=(512,32,24,16), activation='relu', 
                    alpha=0.1, batch_size=256, early_stopping=True, 
                    learning_rate_init=0.003, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=100, tol=1e-8, verbose=True, validation_fraction=0.1, random_state=seed)

print(clf)
clf = BaggingRegressor(clf, n_estimators=10, 
                       max_samples=0.75, verbose=5, n_jobs=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

clf.fit(X_train, y_train)
# there is sudden spike in memory consumption here
import gc
gc.collect()

y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

#gc.collect()

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]

print(p(y_train, y_train_pred))
print(mean_squared_error(y_train, y_train_pred))
print(p(y_test, y_test_pred))
print(mean_squared_error(y_test, y_test_pred))
'''
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
pcc_scorer = make_scorer(p)
scoring = {'mse_scorer':mse_scorer, 'pcc_scorer':pcc_scorer}
#score = cross_val_score(clf, X, y, cv=5, verbose=1, scoring=pcc_scorer)
scores = cross_validate(clf, X, y, cv=10, verbose=5, scoring=scoring)
print(scores)
#print(score)
'''
# grid search

'''
from sklearn.model_selection import GridSearchCV
p_grid = {'learning_rate_init': [0.001],
          'alpha': [0.1, 0.01],
          'batch_size': [256, 512],
          'hidden_layer_sizes': [(8,), (16,), (32,), (64,)]}

gscv = GridSearchCV(clf, param_grid=p_grid, scoring={'Pearson':pcc_scorer,'MSE':mse_scorer}, 
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
