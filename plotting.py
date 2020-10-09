#!/usr/bin/env python
import sys, os, math, random
from random import randrange, randint
from collections import Counter

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def set_properties(ax, x_label, y_label, title=None,
                   x_tick_labels=None, y_tick_labels=None):
    if title is not None:
        ax.set_title(title, fontsize=24)
        
    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    
    tick_label_font_size = 18
    
    if x_tick_labels is None:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_label_font_size)
    else:
        ax.set_xticklabels(x_tick_labels, fontsize=tick_label_font_size)
        
    if y_tick_labels is None:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_label_font_size)
    else:
        ax.set_yticklabels(y_tick_labels, fontsize=tick_label_font_size)
    
def save_base_model_figures(min_size, max_size, last_epoch=30, parent_directory='Base'):
    row_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    
    path = '{}/{}-{}'.format(parent_directory, min_size, max_size)
    metrics = np.load(path + '/Metrics.npy')
    fpr, tpr = np.load('{}/Epoch-{:02d}-ROC.npy'.format(path, last_epoch))
    confusion_matrix = np.load('{}/Epoch-{:02d}-ConfusionMatrix.npy'.format(path, last_epoch))
    print(*zip(row_names, metrics[:, -1]))
    
    fig, (loss_ax, metrics_ax, roc_ax, conf_ax) = plt.subplots(4, figsize=(20, 20 * 4.2))
    
    x_tick_labels = np.arange(metrics.shape[1] + 1)
    loss_ax.plot(metrics[0])
    loss_ax.set_xticks(x_tick_labels)
    print('Loss ticks', loss_ax.get_yticks())
    loss_ax.set_yticks([np.max(metrics[0]), np.min(metrics[0])])
    set_properties(loss_ax, 'Epoch', 'Error',
                   x_tick_labels=x_tick_labels,
                   y_tick_labels=["{:.2g}".format(metrics[0,  0]),
                                  "{:.2g}".format(metrics[0, -1])])
    metr = metrics[1:]
    for m in metr:
        metrics_ax.plot(m)
    
    print(metr[:, -1] * 100)
    metrics_yticks = [0.0, np.min(metr[-2, 0]), metr[1, -1], np.max(metr)]
    metrics_ax.set_xticks(x_tick_labels)
    metrics_ax.set_yticks(metrics_yticks)
    metrics_ax.tick_params(axis='both', which='major', pad=15)
    
    set_properties(metrics_ax, 'Epoch', '',
                   x_tick_labels=x_tick_labels,
                   y_tick_labels=map(lambda val:"{:.2f}%".format(val * 100), metrics_yticks))
    metrics_ax.legend(['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC'], prop={ 'size':20 })
    
    roc_ticks = [0.0, 0.5, 1.0]
    roc_ax.plot(fpr, tpr)
    roc_ax.set_xticks(roc_ticks)
    roc_ax.set_yticks(roc_ticks)
    set_properties(roc_ax, 'False-Positive Rate', 'True-Positive Rate',
                  x_tick_labels=roc_ticks,
                  y_tick_labels=roc_ticks)

    norm_cm = confusion_matrix / np.sum(confusion_matrix, 1, keepdims=True)
    log_norm_cm = np.log(norm_cm + 0.005)
    conf_tick_labels = np.array([i for i, _ in enumerate(range(log_norm_cm.shape[0])) if i % 4 == 0])
    
    sns.heatmap(log_norm_cm - np.min(log_norm_cm),
                cmap="gray_r",
                # annot=True,
                cbar=False,
                fmt='.2g',
                ax=conf_ax,
                xticklabels=conf_tick_labels + 1,
                yticklabels=conf_tick_labels + 1)
    
    conf_ax.set_xticks(conf_tick_labels)
    conf_ax.set_yticks(conf_tick_labels)
    set_properties(conf_ax, 'True Label', 'Prediction')
    
    fig.savefig('metrics.eps', format='eps')
    plt.close(fig)
    

row_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC', 'MCC']
def display_heatmaps(parent_directory, min_size = 5, max_size = 24):
    fig, axes = plt.subplots(len(row_names), figsize=(20, 30 * len(row_names)))
    
    matrix_dim = max_size - min_size + 1
    metric_matrices = np.zeros((7, matrix_dim, matrix_dim))
    
    index_loss = row_names.index('Loss')
    index_roc  = row_names.index('AUROC')
    
    for min_s in range(min_size, max_size + 1):
        for max_s in range(min_s, max_size + 1):
            path = '{}/{}-{}/Metrics.npy'.format(parent_directory, min_s, max_s)
            metrics = np.load(path)

            row    = max_s - min_size
            column = min_s - min_size
            
            metric_matrices[:, row, column] = metrics[:, -1]

    # Font size of annotations
    annot_size_list = [18 for _ in range(len(row_names))]
    annot_size_list[index_loss] = 13
    
    # Number of decimal places to use for each number
    num_decimals_list = [2 for _ in range(len(row_names))]
    num_decimals_list[index_roc] = 3 # Set number of decimals to 3
    
    for corr_matrix, row_name, ax, annot_size, num_decimals in zip(metric_matrices, row_names,
                                                                   axes, annot_size_list,
                                                                   num_decimals_list):
        sns.heatmap(corr_matrix,
                    mask=np.tril(np.ones_like(corr_matrix), k=-1).astype(np.bool).T,
                    cmap="Spectral_r",
                    annot=True, annot_kws = { "size":annot_size },
                    cbar =True,  cbar_kws = { 'use_gridspec':False, 'location':"bottom", 'pad':0.04 },
                    fmt='.{}g'.format(num_decimals),
                    ax=ax,
                    xticklabels=range(min_size, max_size+1),
                    yticklabels=range(min_size, max_size+1))
        
        ax.set_xlabel('Min Motif Size', fontsize=18)
        ax.set_ylabel('Max Motif Size', fontsize=18)
        
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        
        ax.set_title(row_name, fontsize=24)
    
    # increasing font size on colorbar shouldn't be this difficult
    for maybe_colorbar in fig._axstack._elements:
        label = maybe_colorbar[0][1][0][1]
        if label is not None:
            cbar_ax = maybe_colorbar[1][1]
            cbar_ax.set_xticklabels(cbar_ax.get_xticklabels(), fontsize=16)
            cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=16)

    fig.savefig('heatmaps.eps', format='eps')
    plt.close(fig)

if __name__ == "__main__":
    # TODO: folder location arguments. Epochs argument. Min/Max size argument
    save_base_model_figures(5, 24, parent_directory='Base')
    display_heatmaps('Grid-Search', min_size=5, max_size=6)
    
