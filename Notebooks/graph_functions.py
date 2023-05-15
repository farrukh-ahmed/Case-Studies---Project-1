import pandas as pd
import numpy as np
import itertools
import pprint
from helper_functions import read_data, format_variables
# used for the graphs
import seaborn as sns
import os
sns.set(font_scale = 1.2)

# used for plotting
from matplotlib import pyplot as plt
import matplotlib

# setting font to 'Times New Roman'
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 16})


PLOT_COLOR = "blue"


def create_histogram(data_df, col_name, hist_dir_path):
    min_value = data_df[col_name].min()
    max_value = data_df[col_name].max()
    bins = np.linspace(min_value,max_value,20)

    result = plt.hist(data_df[col_name], bins = bins, color=PLOT_COLOR, edgecolor='k', alpha=0.65)
    plt.axvline(data_df[col_name].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(data_df[col_name].median(), color='k', linestyle='dashed', linewidth=2)

    min_ylim, max_ylim = plt.ylim()
    plt.text(data_df[col_name].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(data_df['messwert_bp_sys'].mean()))
    plt.text(data_df[col_name].median()*1.2, max_ylim*0.8, 'Median: {:.2f}'.format(data_df['messwert_bp_sys'].median()))
    plt.xlabel(col_name + ' (mmHg)')
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(hist_dir_path, col_name + "_Hist.pdf"), dpi=180, bbox_inches='tight')
    plt.show()


def create_bar_plot(data_df, col_name, barplot_dir_path):
    print(data_df[col_name].value_counts())
    counts = data_df[col_name].value_counts(ascending=True)
    categories = list(counts.index)
    categories = [str(x) if type(x) == bool else x for x in categories]
    
    figsize = (5, 5)
    if len(categories) > 6:
        figsize = (20, 5)

    f, ax = plt.subplots(figsize=figsize)
    plt.bar(categories, counts, color =PLOT_COLOR,
            width = 0.4)

    plt.xlabel(col_name)
    plt.ylabel("No. of people")
    plt.savefig(os.path.join(barplot_dir_path, col_name + "_Bar.pdf"), dpi=180, bbox_inches='tight')
    plt.show()


def create_heat_map(data_df, cols, misc_dir_path):
    correlation_matrix = data_df[cols].corr()
    fig, ax = plt.subplots(figsize=(15,15))     
    plot = sns.heatmap(correlation_matrix, annot = True, linewidths=.5, ax=ax)
    plt.show()
    plot.figure.savefig(os.path.join(misc_dir_path, "corr_matrix.pdf"), dpi=180, bbox_inches='tight')


def create_scatter_plot(data_df, col_list, x, misc_dir_path):
    plt.figure()
    for i in range(len(col_list)):
        for j in range(i+1, len(col_list)):
            plot = data_df.plot.scatter(col_list[i], col_list[j], label=f'{col_list[i]} vs {col_list[j]}',color=PLOT_COLOR, alpha=0.2)
            plt.xlabel(col_list[i] + " (mmHg)")
            plt.ylabel(col_list[j] + " (mmHg)")
            plt.plot(x, x, color='red')
            plot.get_legend().remove()
            plot.figure.savefig(os.path.join(misc_dir_path, col_list[i] +  "_vs_" + col_list[j] + ".pdf"), bbox_inches='tight')


    plt.show()