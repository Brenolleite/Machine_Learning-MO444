import numpy as np
import linear_regression as linear
import processing as proc

# -------------- Params -------------

# Defines type of linear regression
reg_type = "gradient"

# Defines number of foldes using on traing
n_folds = 3

# Defines learning rate
learning_rate = 0.001

# Defines degree of function used into linear regretion
deg = 1

# Defines verbose flag (debugging mode)
verbose = True

# Defines if should generate graphs
generate_graphs = False

# Defines number of iterations on GD
iterations = 2

model_params = [reg_type, learning_rate, deg, iterations]
# -----------------------------------

# Read training data
train_file = np.loadtxt('../dataset/year-prediction-msd-train.txt', delimiter=',')

# Divide data from labels
train_labels = train_file[:, 0]
train_data = train_file[:, 1:12]

# Pre-prossesing data
train_data = proc.normalize_l2(train_data)

# Training process using K-Fold
models = linear.kfold(model_params, train_data, train_labels, n_folds, verbose, generate_graphs)

# Get best model on the K-Fold training using Mean squared error
best_model = models[models[:, 1][0].argmax()]

if generate_graphs:
    print("generate graphs")

# Reading test file
#test_file = np.loadtxt('year-prediction-msd-test.txt', delimiter=',')

# Divide data from labels
#test_labels = train_file[:, 0]
#test_data = train_file[:, 1:12]

# Pre-prossesing test
#test_data = proc.normalize_l2(test_data)

# Predicting test
#print("Results on Test dataset")
#linear.predict(best_model[0], test_data, test_labels, verbose)