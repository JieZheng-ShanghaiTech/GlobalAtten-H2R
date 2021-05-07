#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 11:58
# @Author  : Eric
# @Site    : 
# @File    : plot.py
# @Software: PyCharm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance(loss, auc, log_file):
    """
    Plot and save loss and accuracy.

    param loss: loss to be plotted
    param accuracy: accuracy to be plotted
    param log_file: target file for storing the resulting plot
    return: None
    """
    loss, auc = pd.DataFrame(loss), pd.DataFrame(auc)

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Binary cross-entropy Loss')

    accuracy_plot = sns.lineplot(data=auc, ax=ax[1])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'AUC')

    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    fig.savefig(log_file)
