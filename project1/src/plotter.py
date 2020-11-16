import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

def make_2plot(x, y1, y2, max, label1="", label2="", xlabel="", ylabel="", title=""):
    plt.semilogy(x, y1, label=label1)
    plt.semilogy(x, y2, label=label2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, max - 1)
    plt.title(title)
    plt.show()



def general_plotter(x_values, y_values, labels="", xlabel="", ylabel="", title="", styles="None"):
    if (isinstance(x_values, (list, tuple, np.ndarray)) and isinstance(y_values, (list, tuple, np.ndarray)) and isinstance(labels, (list, tuple, np.ndarray))):
        for i in range(len(x_values)):
            plt.plot(x_values[i], y_values[i], labels[i])
    else:
        plt.plot(x_values, y_values, labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
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