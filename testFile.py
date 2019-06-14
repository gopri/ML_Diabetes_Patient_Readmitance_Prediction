import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import skfeature as sky
from sklearn.metrics import accuracy_score
import skfeature.function.similarity_based.fisher_score as fs
from sklearn.metrics import accuracy_score

def test(processed_train_features, train_labels, processed_test_features, test_labels):
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split = 20)
    rf_clf.fit(processed_train_features, train_labels)
    rf_test_labels_hat = rf_clf.predict(processed_test_features)
    print("Random forest test accuracy: {}".format(100*accuracy_score(test_labels, rf_test_labels_hat)))

    lr_clf = LogisticRegression(max_iter = 120, C = 2.0, random_state=0)
    lr_clf.fit(processed_train_features, train_labels)
    lr_test_labels_hat = lr_clf.predict(processed_test_features)
    print("Logistic Regression test accuracy: {}".format(100*accuracy_score(test_labels, lr_test_labels_hat)))

    ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate = 1.0, random_state=0)
    ada_clf.fit(processed_train_features, train_labels)
    ada_test_labels_hat = ada_clf.predict(processed_test_features)
    print("Adaboost Classifier test accuracy: {}".format(100*accuracy_score(test_labels, ada_test_labels_hat)))

    nn_clf = MLPClassifier(alpha=0.2,hidden_layer_sizes=(7, 4),random_state=0)
    nn_clf.fit(processed_train_features,train_labels)
    nn_test_labels_hat = nn_clf.predict(processed_test_features)
    print("Neural Net test accuracy: {}".format(100*accuracy_score(test_labels, nn_test_labels_hat)))

