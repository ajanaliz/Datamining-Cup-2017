"""DMC2017, 4/13/17"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from preprocessing import data_preparation

train_full_data = data_preparation.read_data('../data/train.csv')
item_full_data = data_preparation.read_data('../data/items.csv')

print('Data set Loaded!\nTrain Shape: ' + str(train_full_data.shape))

joined_data = pd.merge(train_full_data, item_full_data, on=['pid'])


# Bar plot each feature vs label, n*m subplot
def bar_plot_feature_vs_label(data, label, target, n, m):
    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    counter = 0
    for i in range(0, n):
        for j in range(0, m):
            ax_temp = fig.add_subplot(gs[i, j])
            one_label = data[data[label] == 1][target[counter]].value_counts()
            zero_label = data[data[label] == 0][target[counter]].value_counts()
            df = pd.DataFrame([one_label, zero_label])
            df.index = [label, 'not' + label]
            df.plot(kind='bar', stacked=True, figsize=(15, 8), title='Feature ' + str(target[counter]))
            counter += 1
            if counter >= len(target):
                break
    plt.show()


# Scatter plot each feature vs label, n*m subplot
def scatter_plot_feature_vs_label(data, label, n, m):
    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    label = data[label]
    counter = 0
    for i in range(0, n):
        for j in range(0, m):
            ax_temp = fig.add_subplot(gs[i, j])
            ax_temp.scatter(data[data.columns.values[counter]].values, label)
            ax_temp.title.set_text(str(data.columns.values[counter]))
            counter += 1
            if counter >= len(data.columns.values):
                break
    plt.show()


# Scatter plot each feature vs label, n*m subplot
def hist_plot_feature(data, n, m):
    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    counter = 0
    for i in range(0, n):
        for j in range(0, m):
            ax_temp = fig.add_subplot(gs[i, j])
            ax_temp.hist(data[data.columns.values[counter]].values)
            ax_temp.title.set_text(str(data.columns.values[counter]))
            counter += 1
            if counter >= len(data.columns.values):
                break
    plt.show()


hist_plot_feature(train_full_data.dropna().sample(n=100000), 3, 4)
scatter_plot_feature_vs_label(train_full_data.sample(n=10000), 'revenue', 3, 4)
