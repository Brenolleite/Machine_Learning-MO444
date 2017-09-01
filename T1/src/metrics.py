from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np

def print_errors(mse, mae, r2, evs):
    print("Mean squared error: {0}".format(mse))
    print("Mean Absolute Error: {0}".format(mae))
    print("R2 score: {0}".format(r2))
    print("Explaine Variance Score: {0}\n".format(evs))

def compute_errors(model, predictions, groundTruth, verbose):
    mse = mean_squared_error(groundTruth, predictions)
    r2 = r2_score(groundTruth, predictions)
    mae = mean_absolute_error(groundTruth, predictions)
    evs = explained_variance_score(groundTruth, predictions)
    bias = model.intercept_

    if verbose:
        print_errors(mse, mae, r2, evs)

    return np.array([mse, mae, r2, evs])