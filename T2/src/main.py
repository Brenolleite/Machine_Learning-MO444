# -*- coding: utf-8 -*-
import numpy as np
from numpy import array
import os
import re
from PIL import Image
#from array import array
import linear_regression as linear
import logistic_regression as logistic
import neural_net as net
import processing as proc
import graphs
import keras
from keras.datasets import cifar10
import sys
#from skimage import data, io, filters
from scipy.ndimage.filters import sobel
sys.setrecursionlimit(100000)

# -------------- Params -------------

# Defines type model :
	# "net": neural net
	#"logistic": logistic regression,
	#"multinomial": multinomial logistic regression)
model_type = "net"
n_folds = 2 # Defines number of foldes using on traing
verbose = True # Defines verbose flag (debugging mode)

# --- Regression ---
penalty='l2' 
solver='lbfgs' #melhor pra grandes dados
iterations = 50 # Defines number of iterations 
generate_graphs = False # Defines if should generate graphs

if model_type == "logistic":
    multi_class = 'ovr' # one vs rest (sklearn diz que é usado pra abordagem one vs all, e eu respeito o.o)
else :
    if model_type == "multinomial":
        multi_class = 'multinomial'

# --- Neural Network ---
hidden_layers=1 # 1 or 2
n_neurons_input=3072 # se usar pca, mudar
n_neurons=3072
activation='relu'
final_activation='softmax'
loss='categorical_crossentropy'
optimizer='adadelta'
batch_size=256
epochs=1
n_pca=300
generate_confusionMatrix=False

if model_type == "net":
    # Neural Network
    model_params = [hidden_layers,n_neurons_input, n_neurons, activation, final_activation, loss, optimizer, batch_size, epochs, generate_confusionMatrix]
else :
    # Regression
    model_params = [penalty, solver, multi_class, iterations]

# -----------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train = x_train.astype('float32') / 255. # normalizing
x_test = x_test.astype('float32') / 255.

if model_type == "net":
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

# Pre-prossesing data
#train_data = proc.normalize_l2(train_data) >> ficou ruim
#for i in range(x_train.shape[0]):
#    x_train[i] = sobel(x_train[i]) >> ficou ruim
x_train = proc.st_scale(x_train)
#train_data = proc.ZCA(train_data) >> não consegui fazer funcionar
#x_train = proc.PCA_reduction(x_train, n_pca)


# Training process using K-Fold
if model_type == "net":
    models = net.kfold(model_params, x_train, y_train, n_folds, verbose, generate_graphs)
else :
    models = logistic.kfold(model_params, x_train, y_train, n_folds, verbose, generate_graphs)

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
