import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


def line_plot(file, title, xlabel, ylabel, x, y):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y)
    plt.savefig("../graphs/" + file + '.png')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title("Learning Curve - M/C")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Treinamento")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Treinamento score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validação score")

    plt.legend(loc="best")
    plt.savefig("../graphs/" + "learning" + '.png')