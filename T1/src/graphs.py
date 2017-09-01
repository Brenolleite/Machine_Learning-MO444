import matplotlib.pyplot as plt

def line_plot(file, title, xlabel, ylabel, x, y):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y)
    plt.savefig("../graphs/" + file + '.png')