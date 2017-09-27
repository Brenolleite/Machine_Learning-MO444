from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def print_errors(acc, cm):
    print("Accuracy: {0}".format(acc))
    print("Confusion_Matrix: {0}".format(cm))

def compute_errors(predictions, groundTruth, verbose):
    acc = accuracy_score(groundTruth, predictions)
    cm = confusion_matrix(groundTruth, predictions)

    if verbose:
        print_errors(acc, cm)

    return acc
