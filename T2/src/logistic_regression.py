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
                               max_iter = iterations,
                               warm_start = (iterations == 1),
                               n_jobs = -1)

    return model

def predict(model, data, labels, verbose):
    # Predict data on the validation
    predictions = model.predict(data)

    # Compute metrics
    errors = metrics.compute_errors(model.steps[1][1], predictions, labels, verbose)
    return errors

def kfold(model_params, train_data, train_labels, n_folds, verbose, generate_graphs):
    # Create array for storage models and errors
    models = []

    # If tenerate graphs set iterations on model to 1
    # in order to get cost vs iterations
    if generate_graphs:
        iterations = model_params[3]
        model_params[3] = 1
        steps = []
    else:
        steps = None
        iterations = 1

    # Create KFold validation
    kf = KFold(n_splits = n_folds)
    fold = 0

    for train, validate in kf.split(train_data, train_labels):
        # Create the model using the params
        model = create_model(*model_params)

        if verbose:
            print("================== Fold {0} ==========================".format(fold))
        fold += 1
        
        for i in range(iterations):
            # Train our model using train set
            model.fit(train_data[train], train_labels[train].ravel())

            # Verify on validation set
            score = model.score(train_data[validate], train_labels[validate].ravel())
            #errors = predict(model, train_data[validate], train_labels[validate], verbose)

            if generate_graphs:
                steps.append(errors[0])

        # Store model and erros related to it
        models.append([model, score, np.array(steps)])

        # Reset interations counter
        if generate_graphs:
            steps = []

    # Print avr error
    print("Average errors on {0}-Fold \n============================\n".format(n_folds))
    models = np.array(models)
    score = np.sum(models[:, 1], 0)/n_folds
    metrics.print_acc(score)

    return models
