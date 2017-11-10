# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import cluster as cl
import processing as proc
import metrics

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
#x_train, ncomp = proc.PCA_reduction(x_train, 0.8)

# Using kmeans
labels, centers = cl.k_means(79, x_train)

# Verifying cluster variance
metrics.verify_clusters(labels, 4)

# Create elbow graph
#metrics.elbow_graph(x_train, 0, 200, 1)

# Find medoids
id_medoids, distances = pairwise_distances_argmin_min(centers, x_train)

# Get closest files to medoid (n_closest values)
metrics.closest_docs(x_train, id_medoids, labels, 3)

# Usar no final pra plotar :) ele reduz para duas dimensões para visualizar. aplicar PCA antes ou alguma redução

#TSNE
#tsne = TSNE(n_components=2)

#Y = tsne.fit_transform(x_train)
#plt.figure(figsize=(20, 20))
#plt.plot(Y[:, 1], Y[:, 2], marker='o', color='black', ls='', alpha = 0.5)
#plt.scatter(x_train[:, 0], x_train[:, 1], color='red')
#plt.savefig('tsne2')
