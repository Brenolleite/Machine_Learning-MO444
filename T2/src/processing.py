from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Scale data
def scale(data):
    return scale(data)

# Normalize data
def st_scale(data):
    return StandardScaler(with_mean=True).fit_transform(data)

# Normalize data
def normalize_l2(data):
    return normalize(data, norm = 'l2')

# Reduce Dimensionality using PCA
def PCA_reduction(data, comp):
    return PCA(n_components = comp).fit_transform(data) # whiten=True

# ZCA whitening
def ZCA(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening
