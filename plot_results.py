import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
from sys import argv

# Exp 1: python plot_results.py results-targeted13689-2 results-500reps results-500reps-2
# Exp 3: python plot_results.py compile-errorbar13689-500reps-20pts

# value to plot when encounter np.inf
INF = 1000

def plot_jitter(df, infx, infy, delta=0, highlight_index=None):
    x_values = df.values[:, 0]
    y_values = df.values[:, 1]
    print("Number of values below diagonal: ", np.sum(x_values > y_values))
    print("Number of values above diagonal: ", np.sum(x_values < y_values))
    print("Number of values on diagonal: ", np.sum(x_values == y_values))
    jitter_strength = 0.05
    x_jitter = x_values + jitter_strength * np.random.rand(*x_values.shape)
    y_jitter = y_values + jitter_strength * np.random.rand(*y_values.shape)
    if infy == True and infx == True:
        plot_infx_infy(x_jitter, y_jitter, delta, highlight_index)
    elif infx == True:
        plot_infx(x_jitter, y_jitter, delta, highlight_index)
    else:
        plot_infy(x_jitter, y_jitter, delta, highlight_index)

def plot_normal(df, delta=0, highlight_index=None):
    x_values = df.values[:, 0]
    y_values = df.values[:, 1]
    plot_infx(x_values, y_values, delta, highlight_index)

def plot_infx(x_values, y_values, delta=0, highlight_index=None):
    x_values[x_values == np.inf] = INF
    y_values[y_values == np.inf] = INF
    fig, axs = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [5, 1]})
    (ax1, ax2) = axs
    '''
    Graph:
    ax1 | ax2
    '''
    for ax in axs:
        ax.scatter(x_values, y_values, alpha=0.5)
    # Zoom into subplot
    max_x = max(x_values[x_values != INF])
    xlim = math.ceil(max_x*100)/100
    max_y = max(y_values[y_values != INF])
    ylim = math.ceil(max_y*100)/100
    xlim = max(xlim, ylim)
    ylim = max(xlim, ylim)
    ax1.set_xlim(-0.1, xlim+0.1)
    ax1.set_ylim(-0.1, ylim+0.1)
    ax2.set_xlim(INF-1, INF+1)
    ax2.set_xticklabels(['', '\u221E'])
    # plot y=x line
    diag_x_values = np.linspace(0, 10, 100)
    for ax in axs.flat:
        ax.plot(diag_x_values, diag_x_values, ls='--')
    # highlight specific point in red if given
    if highlight_index:
        for ax in axs:
            ax.scatter(x_values[highlight_index],
                    y_values[highlight_index], color='red', s=50)
    # ax1.plot([0, ax1.get_xlim()[1]], [0, ax1.get_ylim()[1]], ls='--')
    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=False)  # don't put tick labels on the right
    plt.subplots_adjust(wspace=0.15)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([1, 1], [0, 1], **kwargs)
    ax2.plot([0, 0], [0, 1], **kwargs)
    fig.supylabel('Regularized \u03B5')
    fig.supxlabel('Non-regularized \u03B5')

def plot_infy(x_values, y_values, delta=0, highlight_index=None):
    x_values[x_values == np.inf] = INF
    y_values[y_values == np.inf] = INF
    fig, axs = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [1, 5]})
    (ax1, ax2) = axs
    '''
    Graph:
    ax1
    --
    ax2
    '''
    for ax in axs:
        ax.scatter(x_values, y_values, alpha=0.5)
    # Zoom into subplot
    max_x = max(x_values[x_values != INF])
    xlim = math.ceil(max_x*100)/100
    max_y = max(y_values[y_values != INF])
    ylim = math.ceil(max_y*100)/100
    xlim = max(xlim, ylim)
    ylim = max(xlim, ylim)
    ax2.set_xlim(-0.1, xlim+0.1)
    ax2.set_ylim(-0.1, ylim+0.1)
    ax1.set_ylim(INF-1, INF+1)
    ax1.set_yticklabels(['', '\u221E'])
    # plot y=x line
    diag_x_values = np.linspace(0, 10, 100)
    for ax in axs.flat:
        ax.plot(diag_x_values, diag_x_values, ls='--')
    # highlight specific point in red if given
    if highlight_index:
        for ax in axs:
            ax.scatter(x_values[highlight_index],
                    y_values[highlight_index], color='red', s=50)
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)  # don't put tick labels on the right
    ax2.tick_params(labeltop=False)
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_bottom()
    plt.subplots_adjust(wspace=0.15)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    fig.supxlabel('Non-regularized \u03B5')
    fig.supylabel('Regularized \u03B5')

def plot_infx_infy(x_values, y_values, delta=0, highlight_index=None):
    x_values[x_values == np.inf] = INF
    y_values[y_values == np.inf] = INF
    fig, axs = plt.subplots(2, 2, gridspec_kw=dict(width_ratios=[5, 1], height_ratios=[1,5]))
    ((ax1, ax2), (ax3, ax4)) = axs
    '''
    Graph:
    ax1 | ax2
    ---  ---
    ax3 | ax4
    '''
    for ax in axs.flat:
        ax.scatter(x_values, y_values, alpha=0.5)
    max_x = max(x_values[x_values != INF])
    xlim = math.ceil(max_x*100)/100
    max_y = max(y_values[y_values != INF])
    ylim = math.ceil(max_y*100)/100
    xlim = max(xlim, ylim)
    ylim = max(xlim, ylim)

    ax3.set_xlim(-0.1, xlim+0.1)  # most of the data
    ax3.set_ylim(-0.1, ylim+0.1)

    ax4.set_xlim(INF-1, INF+1)
    ax4.set_ylim(-0.1, ylim+0.1)

    ax1.set_xlim(-0.1, xlim+0.1)
    ax1.set_ylim(INF-1, INF+1)

    ax2.set_xlim(INF-1, INF+1)
    ax2.set_ylim(INF-1, INF+1)
    # plot y=x line
    diag_x_values = np.linspace(0, 10, INF)
    for ax in axs.flat:
        ax.plot(diag_x_values, diag_x_values, ls='--')
    # highlight index
    if highlight_index:
        for ax in axs.flat:
            ax.scatter(x_values[highlight_index],
                    y_values[highlight_index], color='red', s=50)
    # hide the spines between axes
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_top()
    ax1.set_yticklabels(['', '\u221E'])
    ax4.set_xticklabels(['', '\u221E'])

    ax2.tick_params(labelright=False)  # don't put tick labels on the right
    ax4.tick_params(labelright=False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)

    plt.subplots_adjust(wspace=0.15)
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([1, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax3.plot([0, 1], [1, 0], transform=ax3.transAxes, **kwargs)
    ax4.plot([1, 0], [1, 0], transform=ax4.transAxes, **kwargs)

    fig.supxlabel('Non-regularized \u03B5')
    fig.supylabel('Regularized \u03B5')    

def plot_epsilons(directories, highlight = False):

    plot_dfs = []
    for directory in directories:
        epsilons = np.loadtxt(f'./{directory}/compiled-epsilons.csv')
        header = np.array(['i', 'reg', 'd0', 'd1e-9', 'd1e-2'])
        df = pd.DataFrame(epsilons, columns=header)
        df = df.sort_values(['i', 'reg'])
        plot_dfs.append(df)

    plot_df = pd.concat(plot_dfs, ignore_index=True)
    print("Plot dataframe")
    # Separate into 2 df for 2 delta values
    df1, df2 = [plot_df.pivot(index='i', columns='reg', values=col)
                     for col in ['d0', 'd1e-9']]
    highlight_index1 = np.where(df1.index == 13689)[0][0] if highlight else None
    highlight_index2 = np.where(df2.index == 13689)[0][0] if highlight else None
    plot_jitter(df1, delta='0', infx= False, infy = True, highlight_index= highlight_index1)
    #plot_jitter(df2, delta='1e-9', infx = False, infy = True, highlight_index= highlight_index2)
    print(df1)
    plt.show()


if __name__ == "__main__":
    if len(argv) < 2:
        print("Please provide a file to proceed.")
        exit(0)
    directories = argv[1:]
    plot_epsilons(directories, highlight=False)