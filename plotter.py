import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15,10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best', 'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

def make_2plot(x, y1, y2, max, label1="", label2="", xlabel="", ylabel="", title=""):
    plt.plot(x, y1, label="Train Error")
    plt.plot(x, y2, label="Test Error")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, max - 1)
    plt.title(title)
    plt.show()


def error_plot(beta, lower, upper):
    yerr = [abs(lower - beta), abs(upper - beta)]
    x = np.arange(1,len(beta) + 1, 1)

    fig, ax = plt.subplots()

    ax.errorbar(x, beta, yerr=yerr, lolims=lower, uplims=upper, linestyle='none', marker='o', elinewidth=2, ecolor="red")
    plt.xticks(np.arange(1, len(beta) + 1, 2))
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\beta_{i}$")
    plt.title(r"Estimators in $\beta$ with corresponding upper/lower bounds")
    plt.show()