import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import metrics


def create_model(type, learning_rate, deg, iterations):
    # Create linear regression
    if type == "gradient":
        # Linear regression using stochastic gradient descent
        model = make_pipeline(PolynomialFeatures(degree = deg),
                              linear_model.SGDRegressor(learning_rate = 'constant',
                                                        eta0          = learning_rate,
                                                        max_iter      = iterations,
                                                        loss          = 'squared_loss',
                                                        warm_start    = (iterations == 1)))
    else:
        # Linear regression using normal equation
        model = make_pipeline(PolynomialFeatures(degree = deg),
                              linear_model.LinearRegression(n_jobs = -1))

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
        iterations = model_params[4]
        model_params[4] = 1
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

        print("================== Fold {0} ==========================".format(fold))
        fold += 1

        for i in range(iterations):
            # Train our model using train set
            model.fit(train_data[train], train_labels[train])

            # Verify on validation set
            errors = predict(model, train_data[validate], train_labels[validate], verbose)

            if generate_graphs:
                steps.append(errors[0])

        # Store model and erros related to it
        models.append([model, errors, steps])


    # Print avr error
    print("Average errors on {0}-Fold \n============================\n".format(n_folds))
    models = np.array(models)
    errors = np.sum(models[:, 1], 0)/n_folds
    metrics.print_errors(*errors)

    return models
