import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import metrics

def create_model(penalty, solver, multi_class, iterations):
    # Create logistic regression
    model = LogisticRegression(penalty = penalty,
                               solver = solver,
                               multi_class= multi_class,
                               max_iter = iterations)

    return model

def kfold(model_params, train_data, train_labels, n_folds, verbose):
    # Create array for storage models and errors
    models = []

    # Create KFold validation
    kf = KFold(n_splits = n_folds)
    fold = 0

    for train, validate in kf.split(train_data, train_labels):
        # Create the model using the params
        model = create_model(*model_params)

        if verbose:
            print("================== Fold {0} ==========================".format(fold))
        fold += 1

        # Train our model using train set
        model.fit(train_data[train], train_labels[train].ravel())

        # Verify on validation set
        score = model.score(train_data[validate], train_labels[validate].ravel())

        # Store model and erros related to it
        models.append([model, score])


    # Print avr error
    print("Average errors on {0}-Fold \n============================\n".format(n_folds))
    models = np.array(models)
    score = np.sum(models[:, 1], 0)/n_folds
    metrics.print_acc(score)

    return models
