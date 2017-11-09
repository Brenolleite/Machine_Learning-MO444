# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt


from PIL import Image
import cluster as cl
import processing as proc
import metrics
#import keras

import cluster as cl
import processing as proc
import metrics

# -------------- Params -------------

# -----------------------------------

# Load dataset
x_train = np.genfromtxt('../documents/data.csv', delimiter=',', skip_header=0, skip_footer=0)

# Load ids
g = open('../documents/ids', 'rb')
y_train = [line.split() for line in g]

# Normalizing
x_train = x_train.astype('float32')

# Pre-prossesing data
x_train = proc.normalize_l2(x_train)
x_train = proc.st_scale(x_train)

# Aplying PCA
x_train, ncomp = proc.PCA_reduction(x_train, 0.8)

#print x_train.shape

# Finding the number of clusters
# 19904 data and 2209 features
range_n_clusters = range(75, 95)
print(range_n_clusters)
#cl.find_clusters(range_n_clusters, x_train)


#calcula elbow
'''
# k means determine k
distortions = []
K = range(5, 105, 5)

for k in K:
    kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(x_train)
    kmeanModel.fit(x_train)
    distortions.append(sum(np.min(cdist(x_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x_train.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('elbow_correto2')
'''

# descobrir centróides
'''
k = 43
kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(x_train)
kmeanModel.fit(x_train)
id_centroids, _ = pairwise_distances_argmin_min(kmeanModel.cluster_centers_, x_train)

print id_centroids
'''

# Usar no final pra plotar :) ele reduz para duas dimensões para visualizar. aplicar PCA antes ou alguma redução

#TSNE
#tsne = TSNE(n_components=2)

#Y = tsne.fit_transform(x_train)
#plt.figure(figsize=(20, 20))
#plt.plot(Y[:, 1], Y[:, 2], marker='o', color='black', ls='', alpha = 0.5)
#plt.scatter(x_train[:, 0], x_train[:, 1], color='red')
#plt.savefig('tsne2')
