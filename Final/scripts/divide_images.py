import numpy as np
import argparse
import os
import shutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def create_image(path, labels, predicts):
    # Defining colors
    right = (65, 247, 19)
    wrong = (255, 25, 25)

    # Writing images
    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", 18)

    # Getting all labels
    pos = 0
    for pred in predicts:
        if pred in labels:
            draw.text((0, pos), pred, fill=right, font=font)
        else:
            draw.text((0, pos), pred, fill=wrong, font=font)

        pos += 18

    return img

def save_image(path, img):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    img.save(path + '.jpg')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labels_path', help='Labels File', required=True)
parser.add_argument('-p', '--predicts_path', help='Predictions File', required=True)
parser.add_argument('-o', '--output_path', help='Output Path', required=True)
parser.add_argument('-d', '--dataset_path', help='Images path', required=True)
parser.add_argument('-t', '--type', help='Type', type=int, required=False)
ARGS = parser.parse_args()

# Removing folder before running experiments
if os.path.exists(ARGS.output_path):
    shutil.rmtree(ARGS.output_path, ignore_errors=True)

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
    path = ARGS.dataset_path + '/' + pred[0] + '.jpg'
    img = create_image(path, ground_truth, pred[1])

    # Go over gt labels comparing each class
    for label in ground_truth:
        # Get path to write output
        path = ARGS.output_path

        # Verify label
        if label in pred[1]:
            path += '/correct/'
        else:
            path += '/wrong/'

        path += label + '/' + pred[0]

        # Save image in the path
        save_image(path, img)

        #save_image(path, img)