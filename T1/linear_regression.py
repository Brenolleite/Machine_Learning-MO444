import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Read training data
train_file = np.loadtxt('year-prediction-msd-train.txt', delimiter=',')

# Need to FIX
train_labels = train_file[:,0]
train_data = train_file[:,1:]

# Create linear regression
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(train_data, train_labels)

# Reading test file
test_file = np.loadtxt('year-prediction-msd-test.txt', delimiter=',')

# Need to FIX
test_gt = test_file[:,0]
test_data = test_file[:,1:]

predictions = reg.predict(test_data)

# The coefficients
print('Coefficients: \n', reg.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(predictions, test_gt))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(predictions, test_gt))
