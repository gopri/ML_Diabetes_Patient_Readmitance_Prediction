from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import skfeature as sky
from sklearn.metrics import accuracy_score
import skfeature.function.similarity_based.fisher_score as fs
import data_loader
def naiveBayes(processed_train_features,processed_valid_features,train_labels,valid_labels,processed_test_features,test_labels):
	model1 = GaussianNB()
	model1.fit(processed_train_features, train_labels)
	naive_bayes_predict_train = model1.predict(processed_train_features)
	naive_bayes_predict_valid = model1.predict(processed_valid_features)
	print("Naive Bayes Training accuracy ",accuracy_score(train_labels, naive_bayes_predict_train))
	print("Naive Bayes Valid accuracy ",accuracy_score(valid_labels, naive_bayes_predict_valid))
	#naive_bayes_predict_train_before_fisher = model1.predict(processed_test_features)
	#print("Naive Bayes Testing accuracy ",accuracy_score(test_labels, naive_bayes_predict_train_before_fisher))
	XFisher = processed_test_features.to_numpy()
	score = fs.fisher_score(XFisher, test_labels)
	ranked_featrues = fs.feature_ranking(score)
	topFeatures = ranked_featrues[:50]
	print(topFeatures)
	print(score.shape)
	print(XFisher.shape)
	intersection_cols = topFeatures
	colnamelist=[]
	for i in topFeatures:
		colname = processed_train_features.columns[i]
		colnamelist.append(colname)
	test = processed_test_features.copy()
	valid_for_bayes = processed_valid_features.copy()
	size = 188
	test.drop(test.columns.difference(colnamelist), 1, inplace=True)
	valid_for_bayes.drop(valid_for_bayes.columns.difference(colnamelist), 1, inplace=True)
	model = GaussianNB()
	model.fit(test, test_labels)
	naive_bayes_predict_train_after_fisher = model.predict(test)
	print("Naive Bayes Testing accuracy ",accuracy_score(test_labels, naive_bayes_predict_train_after_fisher))
	naive_bayes_predict_valid_after_fisher = model.predict(valid_for_bayes)
	print("Naive Bayes Validation accuracy",accuracy_score(valid_labels, naive_bayes_predict_valid_after_fisher))
