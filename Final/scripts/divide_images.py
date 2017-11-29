import numpy as np
import argparse
import os
import cv2

def create_image(labels, predicts):

    return 'image with labels'

def save_image(path, img):
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.write(path, img)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labels_path', help='Labels File', required=True)
parser.add_argument('-p', '--predicts_path', help='Predictions File', required=True)
parser.add_argument('-o', '--output_path', help='Output Path', required=True)
ARGS = parser.parse_args()

# Loading labels file
f = np.genfromtxt(ARGS.labels_path, delimiter=',', dtype=str)

# Removing first line and column
f = f[1:,:]

# Create dict of labels
labels = {}

# Dividing classes
for i in range(len(f)):
    labels[f[i][0]] = f[i][1].split(' ')

# Reading predictions file
f = np.genfromtxt(ARGS.predicts_path, delimiter=',', dtype=str)

# Creating dict of predictions
predicts = {}

# Dividing classes
for i in range(len(f)):
    predicts[f[i][0]] = f[i][1].split(' ')

# Go over all predictions
for pred in predicts.items():
    # Get labels for file
    ground_truth = labels[pred[0]]

    # Create image with labels
    img = create_image(ground_truth, pred[1])

    path = ARGS.output_path

    # Go over gt labels comparing each class
    for label in ground_truth:
        if label in pred[1]:
            path += '/correct/'
        else:
            path += '/wrong/'

        path += label

        #save_image(path, img)