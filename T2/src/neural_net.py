import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import metrics

def create_model(type, learning_rate, deg, iterations):
    # Create neural network entrada de : 32x32x3 = 3072
    if type == "one":
        # Linear regression using stochastic gradient descent
        model = make_pipeline(MLPClassifier(activation='relu',
                                            learning_rate_init=learning_rate,
                                            max_iter=iterations,
                                            momentum=0.9,
                                            solver='sgd',
                                            alpha=1e-5,
                                            hidden_layer_sizes=(1,1000), # layers, neurons
                                            random_state=1))
    else:
        # Linear regression using normal equation
        model = make_pipeline(MLPClassifier(activation='relu',
                                            learning_rate_init=learning_rate,
                                            max_iter=iterations,
                                            momentum=0.9,
                                            solver='sgd',
                                            alpha=1e-5,
                                            hidden_layer_sizes=(2,1000), # layers, neurons
                                            random_state=1))

    return model

def predict(model, data, labels, verbose):
    # Predict data on the validation
    predictions = model.predict(data)

    # Compute metrics
    errors = metrics.compute_errors(predictions, labels, verbose)

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
            errors = predict(model, train_data[validate], train_labels[validate].ravel(), verbose)

            if generate_graphs:
                steps.append(errors[0])

        # Store model and erros related to it
        models.append([model, errors, np.array(steps)])

        # Reset interations counter
        if generate_graphs:
            steps = []

    # Print avr error
    print("Average errors on {0}-Fold \n============================\n".format(n_folds))
    models = np.array(models)
    errors = np.sum(models[:, 1], 0)/n_folds
    metrics.print_errors(*errors)

    return models
