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


X_train = load_npz(sys.argv[1])
X_test = load_npz(sys.argv[3])
print(X_train.shape)
print(X_test.shape)

y_train = load(sys.argv[2])['y']
y_test = load(sys.argv[4])['y']

'''
y_train = np.log(y_train)
y_test = np.log(y_test)

print(np.count_nonzero(np.isnan(y_train)))
print(np.count_nonzero(np.isnan(y_test)))
y_train[np.isnan(y_train)] = 0
y_test[np.isnan(y_test)] = 0
y_train[np.isinf(y_train)] = 0
y_test[np.isinf(y_test)] = 0
'''
#y_train = (y_train - y_train.mean()) / y_train.std()
#y_test = (y_test - y_test.mean()) / y_test.std()
print(y_train.shape)
print(y_test.shape)

if len(sys.argv) == 6:
    seed = int(sys.argv[5])
else:
    seed = None
# data input and preprocessing done


NUMBER_OF_LINEAR_REGRESSOR = 500

def train(X_train, y_train):
	clf = [LinearRegression() for i in range(NUMBER_OF_LINEAR_REGRESSOR)]
	finalClassifier = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=4,
           oob_score=False, random_state=None, verbose=10, warm_start=False)


	y_train_pred_using_individual_nn = np.empty((X_train.shape[0],0), int)

	training_samples_in_each_clf = X_train.shape[0]//NUMBER_OF_LINEAR_REGRESSOR
	
	for i in range(NUMBER_OF_LINEAR_REGRESSOR):
		start = i*training_samples_in_each_clf
		end = (i+1)*training_samples_in_each_clf
		
		clf[i].fit(X_train[start:end], y_train[start:end])
		#print(y_train_pred_using_individual_nn.shape)
		#print(np.array(clf[i].predict(X_train)).transpose().shape)
		y_train_pred_using_individual_nn = np.append(y_train_pred_using_individual_nn, np.array([clf[i].predict(X_train)]).transpose(), axis = 1)
		
		print("{}th linear regressor trained".format(i))
		gc.collect()


	# print(y_train_pred_using_individual_nn)

	print(y_train_pred_using_individual_nn.shape, X_train.shape)

	print("linear regressors trained...........")
	#X_train_final = np.append(y_train_pred_using_individual_nn, X_train, axis = 1)
	#X_train_final = hstack((y_train_pred_using_individual_nn, X_train))
	finalClassifier.fit(y_train_pred_using_individual_nn, y_train)

	#np.savetxt('final_features_train.txt', y_train_pred_using_individual_nn, fmt = '%1.0f')
	# print(X_train_final.shape)
	# clf.fit(X_train, y_train)
	# y_test_pred = clf.predict(X_test)
	# comparePrediction(y_test, y_test_pred)
	# # comparePrediction(y_train, y_train_pred)
	# print(accuracy(y_test, y_test_pred));
	# print(accuracy(y_train, y_train_pred));
	print("training complete...")
	return clf, finalClassifier



def test(clf, finalClassifier, X_test):
	y_test_pred_using_individual_nn = np.empty((X_test.shape[0],0), int)

	for i in range(NUMBER_OF_LINEAR_REGRESSOR):
		#print(y_test_pred_using_individual_nn.shape)
		#print(np.array(clf[i].predict(X_test)).transpose().shape)
		y_test_pred_using_individual_nn = np.append(y_test_pred_using_individual_nn, np.array([clf[i].predict(X_test)]).transpose(), axis = 1)
		
		gc.collect()

	# print(y_train_pred_using_individual_nn)

	#np.savetxt('final_features_test.txt', y_test_pred_using_individual_nn, fmt = '%1.0f')
	X_test_final = hstack((y_test_pred_using_individual_nn, X_test))

	return finalClassifier.predict(y_test_pred_using_individual_nn)



'''
clf = MLPRegressor(hidden_layer_sizes=(768,128,64,32,16,16,8), activation='relu', 
                    alpha=1, batch_size=256, early_stopping=False, 
                    learning_rate_init=0.075, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=200, tol=1e-8, verbose=True, validation_fraction=0.1, random_state=seed)
'''


#clf = RandomForestRegressor(n_estimators=50, n_jobs=4, verbose=10, max_depth=4, max_features='sqrt')

clf, final_clf = train(X_train, y_train)


#clf = Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=1000,
#   normalize=False, random_state=None, solver='auto', tol=1e-5)


'''clf = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=5, max_features='sqrt',
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=5, min_weight_fraction_leaf=0.0,
             n_estimators=25, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
'''
#clf = BaggingRegressor(clf, n_estimators=100, 
#                      max_samples=1.0, verbose=5, n_jobs=2)


#clf.fit(X_train, y_train)
#fi = clf.feature_importances_
#print(np.argsort(fi))
# there is sudden spike in memory consumption here




print(clf[0])
print(final_clf)





gc.collect()

length = len(y_test)
y_test_pred = test(clf, final_clf, X_test)
y_train_pred = test(clf, final_clf, X_train)


print("testing done...................")

#gc.collect()

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]
'''
y_train = np.exp(y_train)
y_test = np.exp(y_test)
y_train_pred = np.exp(y_train_pred)
y_test_pred = np.exp(y_test_pred)
'''
print('Train Pearson Score: {}'.format(p(y_train, y_train_pred)))
print('Train Error: {}'.format(mean_squared_error(y_train, y_train_pred) / 2.0))
#print('Train R2 Score: {}'.format(clf.score(X_train, y_train)))
print('Train Data Mean: {} Standard Deviation: {}'.format(np.mean(y_train), np.std(y_train)))
print('Train Predicted Mean: {} Standard Deviation: {}'.format(np.mean(y_train_pred), np.std(y_train_pred)))

print('Test Pearson Score: {}'.format(p(y_test, y_test_pred)))
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
