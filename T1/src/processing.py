from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA

# Scale data
def scale(data):
    return scale(data)

# Normalize data
def normalize_l2(data):
    return normalize(data, norm = 'l2')

# Reduce Dimensionality using PCA
def PCA_reduction(data, comp):
    return PCA(n_components = comp).fit_transform(data)
