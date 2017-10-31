# -*- coding: utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import cluster as cl
import processing as proc
import metrics
'''
For n_clusters = 5 The average silhouette_score is : -0.00466366
For n_clusters = 20 The average silhouette_score is : -0.0434674
For n_clusters = 50 The average silhouette_score is : -0.0533876
For n_clusters = 100 The average silhouette_score is : -0.0305307
For n_clusters = 200 The average silhouette_score is : -0.0235541

For n_clusters = 3 The average silhouette_score is : 0.00595137
For n_clusters = 4 The average silhouette_score is : -0.00943799
For n_clusters = 6 The average silhouette_score is : -0.00773504
For n_clusters = 12 The average silhouette_score is : -0.0108736



Rodando 1500, 1200, 900, 600, 300
Rodando 10, 25, 30, 35, 40, 45, 55, 60, 65,  70, 80, 90
'''

# -------------- Params -------------

# -----------------------------------

# Load dataset
x_train = np.genfromtxt('../documents/data.csv', delimiter=',', skip_header=10, skip_footer=10)

# Load ids
g = open('../documents/ids', 'rb')
y_train = [line.split() for line in g]

# Normalizing
x_train = x_train.astype('float32') / 255.

# Pre-prossesing data
x_train = proc.normalize_l2(x_train)
x_train = proc.st_scale(x_train)

#print x_train.shape

# Finding the number of clusters
# 19904 data and 2209 features
range_n_clusters = [10, 25, 30, 35, 40, 45, 55, 60, 65, 70, 80, 90]
cl.find_clusters(range_n_clusters, x_train)

# Usar no final pra plotar :) ele reduz para duas dimensões para visualizar. aplicar PCA antes ou alguma redução

#TSNE
#tsne = TSNE(n_components=2)

#Y = tsne.fit_transform(x_train)
#plt.figure(figsize=(20, 20))
#plt.plot(Y[:, 1], Y[:, 2], marker='o', color='black', ls='', alpha = 0.5)
#plt.scatter(x_train[:, 0], x_train[:, 1], color='red')
#plt.savefig('tsne2')