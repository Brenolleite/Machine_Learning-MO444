import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve

# Defines type of linear regression
type = "gradient"

# Defines number of foldes using on traing
n_folds = 10

# Read training data
train_file = np.loadtxt('../dataset/year-prediction-msd-train.txt', delimiter=',')

# Divide data from labels
train_labels = train_file[:, 0]
train_data = train_file[:, 1:12]

train_data = preprocessing.normalize(train_data, norm = 'l2')

# Create linear regression
if type == "gradient":
    # Linear regression using stochastic gradient descent
    reg = linear_model.SGDRegressor(penalty = 'l1', alpha = 0.00001, average = True, max_iter = 5, verbose = False)
else:
    # Linear regression using normal equation
    reg = linear_model.LinearRegression(n_jobs = -1)

print(reg)

# Create KFold validation
kf = KFold(n_splits = n_folds)

for train, validate in kf.split(train_data, train_labels):
    print("================== New Fold ==========================")
    # Train our model using train data
    reg.fit(train_data[train], train_labels[train])

    # Predict data on the validation
    predictions = reg.predict(train_data[validate])

    # Printing metrics
    print("Mean squared error: {0}".format(mean_squared_error(train_labels[validate], predictions)))
    print("R2 score: {0}".format(r2_score(train_labels[validate], predictions)))
    print("Mean Absolute Error: {0}".format(mean_absolute_error(train_labels[validate], predictions)))