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
matplotlib.rcParams.update({'font.size': 12})


PLOT_COLOR = "blue"

UNITS = {
 'schaetzwert_bp_sys': "(mmHg)",
 'schaetzwert_by_dia': "(mmHg)",
 'messwert_bp_sys': "(mmHg)",
 'messwert_bp_dia': "(mmHg)",
 'age': "(years)",
 'month': "",
 'hour': "",
 'day': "",
 'temp': "(°C)",
 'humidity': "(" + "$p/p_{s}$" + ")",
 'temp_min': "(°C)",
 'temp_max': "(°C)",
 'id': "",
 'geburtsjahr': ""
}

def create_histogram(data_df, col_name, hist_dir_path):
    min_value = data_df[col_name].min()
    max_value = data_df[col_name].max()
    bins = np.linspace(min_value,max_value,20)
    f, ax = plt.subplots(figsize=(7, 6))
    result = plt.hist(data_df[col_name], bins = bins, color=PLOT_COLOR, edgecolor='k', alpha=0.65)
    plt.axvline(data_df[col_name].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(data_df[col_name].median(), color='k', linestyle='dashed', linewidth=2)

    min_ylim, max_ylim = plt.ylim()
    plt.text(data_df[col_name].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(data_df['messwert_bp_sys'].mean()))
    plt.text(data_df[col_name].median()*1.2, max_ylim*0.8, 'Median: {:.2f}'.format(data_df['messwert_bp_sys'].median()))
    plt.xlabel(col_name + ' ' + UNITS[col_name])
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(hist_dir_path, col_name + "_Hist.pdf"), dpi=180, bbox_inches='tight')
    plt.show()


def create_bar_plot(data_df, col_name, barplot_dir_path):
    print(data_df[col_name].value_counts())
    counts = data_df[col_name].value_counts(ascending=True)
    categories = list(counts.index)
    categories = [str(x) if type(x) == bool else x for x in categories]
    categories = sorted(categories, key=str.lower)
    
    figsize = (5, 5)
    if len(categories) > 6:
        figsize = (20, 5)

    f, ax = plt.subplots(figsize=figsize)
    plt.bar(categories, counts, color =PLOT_COLOR,
            width = 0.4, alpha=0.65)

    plt.xlabel(col_name)
    plt.ylabel("No. of participants")
    plt.savefig(os.path.join(barplot_dir_path, col_name + "_Bar.pdf"), dpi=180, bbox_inches='tight')
    plt.show()


def create_box_plot(data_df, main_col, group_col, boxplot_dir_path):
    print(round(data_df.groupby(group_col).describe()[main_col], 2))
    f, ax = plt.subplots(figsize=(7, 6))
    order = data_df.groupby(by=[group_col])[main_col].median().sort_values().index
    plot = sns.boxplot(data=data_df, x=group_col, y=main_col, width=.5, orient="v", order=order)
    plot.set(ylabel=main_col + " " + UNITS[main_col])
    plot.set_xticklabels(plot.get_xticklabels(),rotation=45)
    plt.show()
    plot.figure.savefig(os.path.join(boxplot_dir_path, main_col + "_vs_" + group_col + "_boxplot.pdf"), dpi=180)


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
            plt.xlabel(col_list[i] + " " + UNITS[col_list[i]])
            plt.ylabel(col_list[j] + " " + UNITS[col_list[j]])
            plt.plot(x, x, color='red')
            plot.get_legend().remove()
            plot.figure.savefig(os.path.join(misc_dir_path, col_list[i] +  "_vs_" + col_list[j] + ".pdf"), bbox_inches='tight')


    plt.show()


def create_stacked_barplot(data_df, main_col, group_col, stacked_barplot_dir_path):
    main_col_cats = data_df[main_col].unique()
    group_col_cats = data_df[group_col].unique()

    df_rows = []

    for main_col_cat in main_col_cats:
        row = [main_col_cat]
        counts = []

        _sum = 0
        for group_col_cat in group_col_cats:
            count = len(data_df[(data_df[main_col] == main_col_cat) & (data_df[group_col] == group_col_cat)])
            counts.append(count)
            _sum += count
        
        df_rows.append(row + counts)

    for i in range(len(main_col_cats)):
        cat = main_col_cats[i]
        if type(cat) != str:
            main_col_cats[i] = str(cat)
    for i in range(len(group_col_cats)):
        cat = group_col_cats[i]
        if type(cat) != str:
            group_col_cats[i] = str(cat)

    group_col_cats = list(group_col_cats)

    df_counts = pd.DataFrame(df_rows, columns=[main_col] + group_col_cats)

    print(main_col + " vs " + group_col)
    print(df_counts)

    df = df_counts.copy()
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.div(x.sum()).mul(100), axis=1).astype(float)
    ax = df.plot(
    x = main_col,
    rot = 45, 
    kind = 'bar', 
    stacked = True,  
    xlabel=main_col,
    mark_right = True,
    figsize=(10, 8),
    width=1.0,
    ylabel="Percentage of participants",
    fontsize=12
    )
    ax.legend(
    bbox_to_anchor=(1.0, 1.0),
    fontsize='small',
    title=group_col,
    )

    for c in ax.containers:
        label = c.get_label()
        pct_labels = list(df[label])
        count_labels = list(df_counts[label])
        labels = [f'{round(pct, 1)}%' if int(count) > 0 else '' for pct, count in zip(pct_labels, count_labels)]
        ax.bar_label(c, labels=labels, label_type='center', color="black", fontsize=12)

    plt.savefig(os.path.join(stacked_barplot_dir_path, main_col + "_vs_" + group_col + "_stacked_barplot.pdf"), dpi=180)

    return df, df_counts