# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import cluster as cl
import processing as proc
import metrics
import matplotlib.pyplot as plt

# Load dataset
x_train = np.genfromtxt('../documents/data.csv', delimiter=',', skip_header=0, skip_footer=0)

# Load ids
g = open('../documents/ids', 'rb')
y_train = [line.split() for line in g]
g.close()

# Pre-prossesing data
x_train = proc.normalize_l2(x_train)
x_train = proc.st_scale(x_train)

# Aplying PCA
x_train, ncomp = proc.PCA_reduction(x_train, 0.85)

# Set value of K
K = 31

# Using kmeans
labels, centers = cl.k_means(K, x_train)

# Verifying cluster variance
#metrics.verify_clusters(labels, 4)

# Create elbow graph
#metrics.elbow_graph(x_train, 0, 200, 1)

# Find medoids
#id_medoids, distances = pairwise_distances_argmin_min(centers, x_train)

# Get closest files to medoid (n_closest values)
#metrics.closest_docs(x_train, id_medoids, labels, 3)

# TSNE
tsne = TSNE(n_components=2)

Y = tsne.fit_transform(x_train)
plt.figure(figsize=(20, 20))
plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=100, cmap=plt.cm.get_cmap("jet", K), alpha = 0.5)
plt.colorbar(ticks=range(K))
plt.savefig('../output/tsne' + str(K))