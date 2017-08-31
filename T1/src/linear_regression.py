import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve

# Read training data
train_file = np.loadtxt('../dataset/year-prediction-msd-train.txt', delimiter=',')

# ---- PRE-PROCESSING ----
	# Scale data
#data_s = scale(train_file)
	# Normalize data
data_norm = preprocessing.normalize(train_file, norm='l2')
	# Reduce Dimensionality
#pca = PCA(n_components=20)
#data_pca = pca.fit_transform(data_norm)

	# Divide in training and test sets
train_labels = data_norm[:,0]
train_data = data_norm[:,1:]

	# Select some features
#train_labels = train_labels[78:]
#train_data = train_data[78:]

# ---- REGRESSION MODEL ----
	# Create linear regression
reg = linear_model.SGDRegressor(penalty='l1', alpha=0.00001, average=True, n_iter=5, verbose=False)
#reg = linear_model.LinearRegression(n_jobs=-1) # Create linear regression
print(reg)

	# Compute RMSE on training data
reg.fit(train_data,train_labels)
predictions = reg.predict(train_data)
err = predictions-train_labels
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(predictions))

	# Compute RMSE using k-fold cross validation
n_folds = 10
kf = KFold(n_splits=n_folds)
xval_err = 0
t_rmse = np.array([])
cv_rmse = np.array([])
#fig, ax = plt.subplots()

for train, test in kf.split(train_data):
    reg.fit(train_data[train], train_labels[train])
    predictions = reg.predict(train_data[test])
    err = predictions - train_labels[test]
    xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(train_data))
    #print('Coefficients:', reg.coef_) # The coefficients
    print("Mean squared error: %.2f" % mean_squared_error(predictions, train_labels[test])) # The mean squared error
    # Explained variance score: 1 is perfect prediction
    r2_train = reg.score(train_data[train], train_labels[train])
    r2_test = reg.score(train_data[test], train_labels[test])
    print('R2 no set de treino: %.2f' % r2_train)
    print('R2 no set de teste: %.2f' % r2_test)
    print('Variance score: %.2f' % r2_score(predictions, train_labels[test]))
    print('slope = {0}'.format(reg.coef_[0]))
    print('intercept(bias) = {0}'.format(reg.intercept_))
    plt.scatter(train_data[test][:,0], train_labels[test],  color='red')
    plt.plot(train_data[test][:,0], predictions, color='blue', linewidth=1)


t_rmse = np.append(t_rmse, [rmse_train])
cv_rmse = np.append(cv_rmse, [rmse_10cv])
print('RMSE treino {:.4f}\t\t RMSE 10CV {:.4f}'.format(rmse_train,rmse_10cv))

	# Plot Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=reg, X=train_data, y=train_labels, cv=n_folds, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring="neg_mean_squared_error")

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
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")
#tr_err = np.mean(train_errs, axis=1)
#cv_err = np.mean(cv_errs, axis=1)

#fig, ax = plt.subplots()
#ax.plot(train_sz, tr_err, linestyle="--", color="r", label="training error")
#ax.plot(train_sz, cv_err, linestyle="-", color="b", label="cv error")
#ax.legend(loc="lower right")
plt.show()

	# Plot outputs
plt.scatter(train_data[test][:,0], train_labels[test],  color='red')
plt.scatter(train_data[test][:,0], predictions, color='blue', linewidth=1)
plt.show()

# ---- TEST ----
# Reading test file
#test_file = np.loadtxt('../dataset/year-prediction-msd-test.txt', delimiter=',')

#data_norm_test = preprocessing.normalize(test_file, norm='l2')

# Need to FIX
#test_gt = data_norm_test[:,0]
#test_data = data_norm_test[:,1:]

#predictions = reg.predict(test_data)

# The coefficients
#print('--- Test Set')
#print('Coefficients: \n', reg.coef_)

# The mean squared error
#print("Mean squared error: %.2f"      % mean_squared_error(predictions, test_gt))

# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(predictions, test_gt))
#r2_train = reg.score(train_data, train_labels)
#r2_test = reg.score(test_data, test_gt)
#print('R2 no set de treino: %.2f' % r2_train)
#print('R2 no set de teste: %.2f' % r2_test)
