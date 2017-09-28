# -*- coding: utf-8 -*-
import numpy as np
from numpy import array
import os
import re
from PIL import Image
#from array import array
import linear_regression as linear
import neural_net as net
import processing as proc
import graphs
from keras.datasets import cifar10
import sys
from skimage import data, io, filters
sys.setrecursionlimit(100000)

# -------------- Params -------------

# Defines type of linear regression / size of hidden layers (one or two)
reg_type = "two"

# Defines number of foldes using on traing
n_folds = 10

# Defines learning rate
learning_rate = 0.001

# Defines degree of function used into linear regretion
deg = 1

# Defines verbose flag (debugging mode)
verbose = True

# Defines if should generate graphs
generate_graphs = False

# Defines number of iterations on GD
iterations = 200

# neural net
#solver='sgd'

model_params = [reg_type, learning_rate, deg, iterations]
# -----------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train = x_train.astype('float32') / 255. # normalização
#x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print x_train.shape
print x_test.shape


train_data = x_train
train_labels = y_train

# Pre-prossesing data
#train_data = proc.normalize_l2(train_data) >> ficou ruim
#train_data = filters.sobel(train_data) >> ficou ruim
train_data = proc.st_scale(train_data)
#train_data = proc.ZCA(train_data) >> não consegui fazer funcionar
train_data = proc.PCA_reduction(train_data, 300)

# Training process using K-Fold
models = net.kfold(model_params, train_data, train_labels, n_folds, verbose, generate_graphs)

# Get best model on the K-Fold training using Mean squared error
best_model = models[models[:, 1][0].argmax()]

#if generate_graphs:
    # learning curve
    #graphs.plot_learning_curve(best_model[0].steps[1][1], "TESTE", train_data, train_labels)

    # Generating cost vs iterations
#    costs = best_model[2]
#    iterations = np.arange(costs.shape[0]) + 1
#    graphs.line_plot("CostXInteractions", "Custo vs Iteracoes", "Iteracoes", "Custo", iterations, costs)


# Pre-prossesing test
#test_data = proc.normalize_l2(test_data)

# Predicting test
#print("Results on Test dataset")
#linear.predict(best_model[0], test_data, test_labels, verbose)
