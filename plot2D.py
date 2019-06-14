import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot2DMat(matrix, X, Y, Xtitle, Ytitle, title):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    cax1 = ax.matshow(matrix, interpolation = "nearest")
    fig.colorbar(cax1)
    ax.set_xticklabels([''] + list(X))
    ax.set_yticklabels([''] + list(Y))
    ax.set_title(title)
    ax.legend()
    ax.set(xlabel=Xtitle, ylabel=Ytitle)
    plt.show()