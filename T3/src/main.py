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


# -------------- Params ------------------------
# Set value of K
K = 2

# Generate elbow graph
elbow = False

# Print closest docs from medoid
closest_docs = 3

# Use PCA with percentage (0 - 1)
PCA = False

# Verify cluster variance and histograms (N cluster)
n_cluster = 0

# Generate TSNE graph
TSNE_graph = True

# -------------- Params ------------------------

# Load dataset
x_train = np.genfromtxt('../documents/data.csv', delimiter=',', skip_header=0, skip_footer=19500)

# Load ids
g = open('../documents/ids', 'rb')
y_train = [line.split() for line in g]
g.close()

# Pre-prossesing data
x_train = proc.normalize_l2(x_train)
x_train = proc.st_scale(x_train)

if PCA:
    # Aplying PCA
    print('==> Applying PCA')
    x_train, ncomp = proc.PCA_reduction(x_train, PCA)

# Using kmeans
print('==> Applying K-Means')
labels, centers = cl.k_means(K, x_train)

if n_cluster:
    print("==> Cluster metrics")
    # Verifying cluster variance
    metrics.verify_clusters(labels, n_cluster)

if elbow:
    print("==> Generating elbow graph")
    # Create elbow graph (X, start, end, step)
    metrics.elbow_graph(x_train, 10, 100, 5)

if closest_docs:
    print("==> Getting closest documents")

    # Find medoids
    id_medoids, distances = pairwise_distances_argmin_min(centers, x_train)

    print(centers, distances, id_medoids)

    # Get closest files to medoid (n_closest values)
    #metrics.closest_docs(x_train, id_medoids, labels, 3)

# TSNE
if TSNE_graph:
    print("==> Generating t-SNE graph")
    tsne = TSNE(n_components=2)

    Y = tsne.fit_transform(x_train)
    plt.figure(figsize=(20, 20))
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=100, cmap=plt.cm.get_cmap("jet", K), alpha = 0.5)
    plt.colorbar(ticks=range(K))
    plt.savefig('../output/tsne' + str(K))
