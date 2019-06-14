
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import data_loader
def rf_randomSearch(processed_train_features,train_labels, processed_valid_features,valid_labels):
	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	n_estimators= [10]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth = [3]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
	print(random_grid)
	rf = RandomForestRegressor()
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)


	rf_random.fit(processed_train_features, train_labels)
	train_predict = rf_random.predict(processed_train_features)

	errors = abs(train_predict - train_labels)
	mape = 100 * np.mean(errors)
	print(mape)
	accuracy = 100 - mape
	print("Accuracy ",accuracy)


	rf_random.fit(processed_valid_features, valid_labels)
	valid_predict = rf_random.predict(processed_valid_features)
	errors = abs(valid_predict - valid_labels)
	mape = 100 * np.mean(errors)
	#print(mape)
	accuracy = 100 - mape
	print("Accuracy ",accuracy)


#notused
def lr_param_selection(clf_lr_hp, processed_train_features, train_labels, nfolds=3):
    dual=[True,False]
    max_iter=[100,110,120,130,140]
    C = [1.0,1.5,2.0,2.5]
    param_grid = dict(dual=dual,max_iter=max_iter,C=C)
    random = RandomizedSearchCV(estimator=clf_lr_hp, param_distributions=param_grid, cv = 3, n_jobs=-1)
    random_result = random.fit(processed_train_features, train_labels)
    print("Best parameters: {}".format(random_result.best_params_))
    return random_result.best_params_

#notused
def svc_param_selection(clf_svm_hp, processed_train_features, train_labels, nfolds=3):
    kernels = ['rbf', 'poly']
    gammas = [0.001, 0.01, 0.1, 1]
    Cs = [0.001, 0.01, 0.1, 1, 10]
    param_grid = dict(kernel=kernels, gamma=gammas, C=Cs)
    random = RandomizedSearchCV(estimator=clf_svm_hp, param_distributions=param_grid, cv = 3, n_jobs=-1)
    random_result = random.fit(processed_train_features, train_labels)
    print("Best parameters: {}".format(random_result.best_params_))
    return random_result.best_params_
