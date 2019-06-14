import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
n_estimators = [10,30,50,70,100]
max_depths = [2,5,10,12,13,15,20]

rf_train_accuracy_res = []
rf_valid_accuracy_res = []
#rf_test_accuracy_res = []

rf_train_labels_hat_res = []
rf_valid_labels_hat_res = []
#rf_test_labels_hat_res = []

def rfPlots():
    # Plot
    fig, ax = plt.subplots(1,3, figsize=(15,3))
    ax[0].plot(max_depths, rf_train_accuracy_res[4], label = "Train dataset")
    ax[0].plot(max_depths, rf_valid_accuracy_res[4], label = "Valid dataset")
    ax[0].set_title("Random forest Accuracy for 100 estimators")

    ax[1].plot(max_depths, rf_train_accuracy_res[2], label = "Train dataset")
    ax[1].plot(max_depths, rf_valid_accuracy_res[2], label = "Valid dataset")
    ax[1].set_title("Random forest Accuracy for 50 estimators")

    ax[2].plot(max_depths, rf_train_accuracy_res[0], label = "Train dataset")
    ax[2].plot(max_depths, rf_valid_accuracy_res[0], label = "Valid dataset")
    ax[2].set_title("Random forest Accuracy for 10 estimators")

    for axes in ax.flat:
        axes.set(xlabel='Max_depth', ylabel='Accuracy')
        axes.legend()

    # Plot
    fig, ax = plt.subplots(1,2, figsize=(15,3))
    ax[0].plot(max_depths, rf_train_accuracy_res[4], label = "100 estimators")
    ax[0].plot(max_depths, rf_train_accuracy_res[2], label = "50 estimators")
    ax[0].plot(max_depths, rf_train_accuracy_res[0], label = "10 estimators")
    ax[0].set_title("Random forest Accuracy on train dataset")

    ax[1].plot(max_depths, rf_valid_accuracy_res[4], label = "100 estimators")
    ax[1].plot(max_depths, rf_valid_accuracy_res[2], label = "50 estimators")
    ax[1].plot(max_depths, rf_valid_accuracy_res[0], label = "10 estimators")
    ax[1].set_title("Random forest Accuracy on validation dataset")

    for axes in ax.flat:
        axes.set(xlabel='Max_depth', ylabel='Accuracy')
        axes.legend()

def randomForest(processed_train_features, train_labels, processed_valid_features,valid_labels):
    for i,n_estimator in enumerate(n_estimators):
        rf_train_accuracy_res.append([])
        rf_valid_accuracy_res.append([])
        
        rf_train_labels_hat_res.append([])
        rf_valid_labels_hat_res.append([])
        
        for j,max_depth in enumerate(max_depths):
            print("\nn_estimator: {}, max_depth: {} ::".format(n_estimator, max_depth))
            clf_rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, min_samples_split = 12)
            clf_rf.fit(processed_train_features, train_labels)

            rf_train_labels_hat = clf_rf.predict(processed_train_features)
            train_acc = 100 * accuracy_score(train_labels, rf_train_labels_hat)
            print("Train Accuracy: {}".format(train_acc))
            rf_train_accuracy_res[i].append(train_acc)
            rf_train_labels_hat_res[i].append(rf_train_labels_hat)
                
            rf_valid_labels_hat = clf_rf.predict(processed_valid_features)
            valid_acc = 100 * accuracy_score(valid_labels, rf_valid_labels_hat)
            print("Validation Accuracy: {}".format(valid_acc))
            rf_valid_accuracy_res[i].append(valid_acc)
            rf_valid_labels_hat_res[i].append(rf_valid_labels_hat)

    print("\nRandom Forest Train Accuracy: {}".format(rf_train_accuracy_res))
    print("\nRandom Forest Validation Accuracy: {}".format(rf_valid_accuracy_res))

    plot2D.plot2DMat(rf_train_accuracy_res, max_depths, n_estimators, 'max_depth', 'n_estimator', 'Training Accuracy for random forest')
    plot2D.plot2DMat(rf_valid_accuracy_res, max_depths, n_estimators, 'max_depth', 'n_estimator', 'Validation Accuracy for random forest')

    rfPlots()