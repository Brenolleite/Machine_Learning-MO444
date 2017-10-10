# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import logistic_regression as logistic
import neural_net as net
import processing as proc
import metrics
import keras
from keras.datasets import cifar10
import sys
#from skimage import data, io, filters
from scipy.ndimage.filters import sobel
sys.setrecursionlimit(100000)

# -------------- Params -------------

training = False # Set for k-fold training

# Defines type model (model_type):
	#  "net"		    : Neural Net
	#  "logistic"		: Logistic Regression,
	#  "multinomial"	: Multinomial Logistic Regression)
model_type =    "net"
n_folds = 		5 	    # Defines number of foldes using on traing
verbose = 		True 	# Defines verbose flag (debugging mode)

# --- Regression ---
penalty =		    'l2'
solver = 		    'lbfgs'
iterations = 		50 	        # Defines number of iterations
generate_graphs = 	False 	    # Defines if should generate graphs

if model_type == "logistic":
    multi_class = 'ovr' # one vs rest
else :
    if model_type == "multinomial":
        multi_class = 'multinomial'

# --- Neural Network Config---
hidden_layers    =  2
n_neurons_input  = 	3072
n_neurons        =	3800
activation       =	'relu'
final_activation = 	'softmax'
loss             =	'categorical_crossentropy'
optimizer        =  'adadelta'
batch_size       =  256
epochs           =	20
n_pca            =	500
confusionMatrix  =  False

if model_type == "net":
    # Neural Network
    model_params = [hidden_layers,n_neurons_input, n_neurons, activation, final_activation, loss, optimizer, batch_size, epochs, confusionMatrix]
else :
    # Regression
    model_params = [penalty, solver, multi_class, iterations]

# -----------------------------------

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Normalizing
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

if model_type == "net":
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

# Pre-prossesing data
x_train = proc.normalize_l2(x_train)

#for i in range(x_train.shape[0]):
#   x_train[i] = sobel(x_train[i])

x_train = proc.st_scale(x_train)

# train_data = proc.ZCA(train_data)
# x_train = proc.PCA_reduction(x_train, n_pca)

# Trainning process using K-Fold
if training:
    if model_type == "net":
        models = net.kfold(model_params, x_train, y_train, n_folds, verbose)
    else :
        models = logistic.kfold(model_params, x_train, y_train, n_folds, verbose)

if not training:
    # Pre-prossesing test
    x_test = proc.normalize_l2(x_test)
    x_test = proc.st_scale(x_test)

    if model_type == "net":
        # Testing process on neural net
        net.test(model_params, x_train, y_train, x_test, y_test)
    else:
        print("Testing Logistic Regression on Test dataset")
         # Testing process on logistic regression
        model = logistic.create_model(*model_params)
        model.fit(x_train, y_train.ravel())
        score = model.score(x_test, y_test.ravel())
        metrics.print_acc(score)
