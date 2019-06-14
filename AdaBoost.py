import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot2D

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

n_estimators=[20, 50, 100, 120, 150, 200]
learning_rates = [0.2, 0.5, 0.8, 1.0, 1.5]

ada_train_accuracy_res = []
ada_valid_accuracy_res = []
ada_test_accuracy_res = []

ada_train_labels_hat_res = []
ada_valid_labels_hat_res = []
ada_test_labels_hat_res = []

def abPlots():
    # Plot
    fig, ax = plt.subplots(1,2, figsize=(15,3))
    ax[0].plot(learning_rates, ada_train_accuracy_res[5], label = "200 n_estimators")
    ax[0].plot(learning_rates, ada_train_accuracy_res[2], label = "100 n_estimators")
    ax[0].plot(learning_rates, ada_train_accuracy_res[0], label = "20 n_estimators")
    ax[0].set_title("AdaBoost Classifier Accuracy on train dataset")

    ax[1].plot(learning_rates, ada_valid_accuracy_res[5], label = "200 n_estimators")
    ax[1].plot(learning_rates, ada_valid_accuracy_res[2], label = "100 n_estimators")
    ax[1].plot(learning_rates, ada_valid_accuracy_res[0], label = "20 n_estimators")
    ax[1].set_title("AdaBoost Classifier Accuracy on validation dataset")

    for axes in ax.flat:
        axes.set(xlabel='learing rate', ylabel='Accuracy')
        axes.legend()

def adaBoost(processed_train_features, train_labels, processed_valid_features,valid_labels):
    for i, n_estimator in enumerate(n_estimators):
        ada_train_accuracy_res.append([])
        ada_valid_accuracy_res.append([])
        ada_test_accuracy_res.append([])

        ada_train_labels_hat_res.append([])
        ada_valid_labels_hat_res.append([])
        ada_test_labels_hat_res.append([])
        
        for j, learning_rate in enumerate(learning_rates):
            print("\nn_estimators: {}, learning_rate: {} ::".format(n_estimator, learning_rate))
            clf_ada = AdaBoostClassifier(n_estimators=n_estimator, learning_rate = learning_rate, random_state=0)
            clf_ada.fit(processed_train_features, train_labels)

            ada_train_labels_hat = clf_ada.predict(processed_train_features)
            train_acc = 100 * accuracy_score(train_labels, ada_train_labels_hat)
            print("Train Accuracy: {}".format(train_acc))
            ada_train_accuracy_res[i].append(train_acc)
            ada_train_labels_hat_res[i].append(ada_train_labels_hat)
                
            ada_valid_labels_hat = clf_ada.predict(processed_valid_features)
            valid_acc = 100 * accuracy_score(valid_labels, ada_valid_labels_hat)
            print("Validation Accuracy: {}".format(valid_acc))
            ada_valid_accuracy_res[i].append(valid_acc)
            ada_valid_labels_hat_res[i].append(ada_valid_labels_hat)
            
    print("\nAdaBoost Train Accuracy: {}".format(ada_train_accuracy_res))
    print("\nAdaBoost Validation Accuracy: {}".format(ada_valid_accuracy_res))

    plot2D.plot2DMat(ada_train_accuracy_res, learning_rates, n_estimators, 'learning_rate', 'n_estimators', "Training Accuracy for AdaBoost Classifier")
    plot2D.plot2DMat(ada_valid_accuracy_res, learning_rates, n_estimators, 'learning_rate', 'n_estimators', "Validation Accuracy for AdaBoost Classifier")

    abPlots()