import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_loader
import plot2D

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def NeuralNets(processed_train_features,train_labels, processed_valid_features,valid_labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(processed_train_features,train_labels)
    y_train = clf.predict(processed_train_features)
    y_valid = clf.predict(processed_valid_features)
    print("Neural Nets Training accuracy ",accuracy_score(train_labels,y_train ))
    print("Neural Nets Validation accuracy ",accuracy_score(valid_labels, y_valid))
	#learning_rate=0.01
    train_list = np.zeros((5, 5))
    valid_list = np.zeros((5, 5))
    learning_rate = np.linspace(0.01,0.05,5)
    hidden_layer_sizes = [7,9,11,13,15]
    #learning_rate = [0.1,0.2,0.3]
    for i in range(0,5):
        for j in range(0,5):
            clf1 = MLPClassifier(alpha=learning_rate[i],hidden_layer_sizes=(i+5, j+2),random_state=0)
            clf1.fit(processed_train_features,train_labels)
            y_train = clf1.predict(processed_train_features)
            y_valid = clf1.predict(processed_valid_features)
            train_list[i][j] = accuracy_score(train_labels,y_train)
            valid_list[i][j]= accuracy_score(valid_labels, y_valid)
    
    print("\nNeural Network Train Accuracy: {}".format(train_list))
    print("\nNeural Network Validation Accuracy: {}".format(valid_list))
    plot2D.plot2DMat(train_list*100, hidden_layer_sizes, learning_rate, "Hidden Layer Size", "Learning Rate", "Train Accuracy for neural Net")
    plot2D.plot2DMat(valid_list*100, hidden_layer_sizes, learning_rate, "Hidden Layer Size", "Learning Rate", "Validation Accuracy for neural Net")