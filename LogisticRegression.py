import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot2D


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

max_iters=[20, 50, 100, 120, 150, 200]
Cs = [1.0,1.5,2.0,2.5]
#max_iters = np.linspace(10,400, num = 20)
#Cs = np.linspace(0.01, 2.0, num = 10)

lr_train_accuracy_res = []
lr_valid_accuracy_res = []

lr_train_labels_hat_res = []
lr_valid_labels_hat_res = []

def lrPlots():
        # Plot
    Cs = [1.0,1.5,2.0,2.5]
    fig, ax = plt.subplots(1,2, figsize=(15,3))
    ax[0].plot(Cs, lr_train_accuracy_res[5], label = "200 max_iters")
    ax[0].plot(Cs, lr_train_accuracy_res[2], label = "100 max_iters")
    ax[0].plot(Cs, lr_train_accuracy_res[0], label = "20 max_iters")
    ax[0].set_title("Logistic Regression Accuracy on train dataset")

    ax[1].plot(Cs, lr_valid_accuracy_res[5], label = "200 max_iters")
    ax[1].plot(Cs, lr_valid_accuracy_res[2], label = "100 max_iters")
    ax[1].plot(Cs, lr_valid_accuracy_res[0], label = "20 max_iters")
    ax[1].set_title("Logistic Regression Accuracy on validation dataset")

    for axes in ax.flat:
        axes.set(xlabel='C (inverse of regularization strength)', ylabel='Accuracy')
        axes.legend()

def logisticRegression(processed_train_features, train_labels, processed_valid_features,valid_labels):
    for i, max_iter in enumerate(max_iters):
        lr_train_accuracy_res.append([])
        lr_valid_accuracy_res.append([])

        lr_train_labels_hat_res.append([])
        lr_valid_labels_hat_res.append([])
        
        for j, C in enumerate(Cs):
            print("\nMax_iter: {}, C: {} ::".format(max_iter, C))
            clf_lr = LogisticRegression(C = C, max_iter= max_iter, random_state=0)
            clf_lr.fit(processed_train_features, train_labels)

            lr_train_labels_hat = clf_lr.predict(processed_train_features)
            train_acc = 100 * accuracy_score(train_labels, lr_train_labels_hat)
            print("Train Accuracy: {}".format(train_acc))
            lr_train_accuracy_res[i].append(train_acc)
            lr_train_labels_hat_res[i].append(lr_train_labels_hat)
                
            lr_valid_labels_hat = clf_lr.predict(processed_valid_features)
            valid_acc = 100 * accuracy_score(valid_labels, lr_valid_labels_hat)
            print("Validation Accuracy: {}".format(valid_acc))
            lr_valid_accuracy_res[i].append(valid_acc)
            lr_valid_labels_hat_res[i].append(lr_valid_labels_hat)
            
    print("\nLogistic Regression Train Accuracy: {}".format(lr_train_accuracy_res))
    print("\nLogistic Regression Validation Accuracy: {}".format(lr_valid_accuracy_res))

    plot2D.plot2DMat(lr_train_accuracy_res, Cs, max_iters, 'C', 'max_iter', "Training Accuracy for Logistic Regression")
    plot2D.plot2DMat(lr_valid_accuracy_res, Cs, max_iters, 'C', 'max_iter', "Validation Accuracy for Logistic Regression")

    lrPlots()