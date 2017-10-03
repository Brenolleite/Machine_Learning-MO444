from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def print_acc(acc):
    print("Accuracy: {0}".format(acc))

def print_confusionMatrix(cm):
    print("Confusion_Matrix: {0}".format(cm))

def compute_acc(predictions, groundTruth, verbose):
    acc = accuracy_score(groundTruth, predictions)

    if verbose:
        print_acc(acc)

    return acc

def print_acc_nn(score):
    print("Test loss: {0}", score[0])
    print("Test accuracy: {0}", score[1])

def confusionMatrix (predictions, groundTruth):
    cm = confusion_matrix(groundTruth, predictions)
    print_confusionMatrix(cm) # gerar grafico bonitinho
