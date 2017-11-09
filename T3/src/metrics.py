import numpy as np
import re
import cluster as cl
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import paired_distances
import matplotlib.pyplot as plt


# Verify variance for clusters, and create histogram of classes
def verify_clusters(labels, qtde = None):
    # Find clusters
    clusters = np.unique(labels)

    # Create list of newsgroups
    f = open('newsgroups.data', 'r')
    newsgroups = [line.rstrip('\n') for line in f]
    f.close()

    # Read ids of files
    f = open('../documents/ids', 'r')
    ids = [line.rstrip('\n') for line in f]
    f.close()

    # Create array of histograms
    histograms = []

    # Create normalization factor
    factor = 0

    for cluster in clusters:
        # Create a new histogram
        hist = np.zeros(len(newsgroups))

        # Create dictionary of newsgroups
        hist = dict(zip(newsgroups, hist))

        # Get rows of files (idx on ids)
        files_idx = np.where(labels == cluster)[0]

        # Update factor for normalization
        if len(files_idx) > factor:
            factor = len(files_idx)

        # Run over all files
        for idx in files_idx:
            # Open and read file
            f = open('../documents/docs/' + ids[idx])
            filetext = filetext = f.read()
            f.close()

            # Use regex to find newsgroups for each text file
            # Fixing issues where a newsgroups is shown on the message
            matches = re.findall('Newsgroups: [\\/\w.,\d-]+', filetext)

            # Go over all matches
            for match in matches:
                # Get all groups for that file
                groups = match.replace('Newsgroups: ', '').split(',')

                # Counting the occurrence of each group
                for item in groups:
                    # Fix issue in the files
                    # 'b04aa255198b5e2526cff7c76c7c6257ad70e49f'
                    # '1cfd267dfba20241fac4126124d73c27840c27fa'
                    # Where there is a comma but not another group
                    if item != '' and item != 'm.h.a':
                        if item in newsgroups:
                            hist[item] += 1
                        else:
                            hist[item] = 1

        # Sort newsgroups
        hist = sorted(hist.items(), reverse=True, key=lambda x: x[1])

        # Adding histogram and count of classes to array
        histograms.append((hist, 0))

    # Generating variance, and organizing bins
    ret_histograms = []
    for hist in histograms:
        # Normalizing
        array = np.array(hist[0])[:,1].astype(np.float)
        for i in range(len(array)):
            array[i] = array[i]/factor

        # Getting variance value
        var = np.var(array)

        # Removing small values from dict
        hist = [i for i in hist[0] if i[1] != 0]

        ret_histograms.append((hist, var))

    for i in range(len(ret_histograms)):
        print('Cluster: {0} -> Variance: {1}'.format(i, ret_histograms[i][1]))

    # Create histogram with qtde of bins
    if qtde != None:
        for i in range(len(ret_histograms)):
            print('Cluster: {0} -> {1}'.format(i, np.array(ret_histograms)[i,0][0:qtde]))

# kmeans generate elbow graph to check value of K
def elbow_graph(x_train, start, end, step):
    distortions = []
    K = range(start, end, step)

    # Try out some K values
    for k in K:
        _ , centroids = cl.k_means(k, x_train)
        distortions.append(sum(np.min(cdist(x_train, centroids, 'euclidean'), axis=1)) / x_train.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('elbow')

def closest_docs(x_train, id_medoids, labels, n_closest):
    # Read ids of files
    f = open('../documents/ids', 'r')
    ids = [line.rstrip('\n') for line in f]
    f.close()

    for idx in range(len(id_medoids)):
        # Get medoid ID
        medoid_id = id_medoids[idx]

        # Get train id from elements from cluster
        cluster_ids = list(np.arange(len(x_train)))

        # Remove medoid from list of ids
        if len(cluster_ids) > 1:
            cluster_ids.remove(medoid_id)

        # Get all data using cluster ids - medoid_id
        X = x_train[cluster_ids]

        # Create a matrix of medoids to make distance measurement
        medoids = np.repeat(x_train[medoid_id], len(X)).reshape(X.shape)

        # Get distance from medoid matrix and data
        distances = paired_distances(medoids, X, 'euclidean')

        # Create indexes for the list of distances
        distances = {v: k for v, k in enumerate(distances)}

        # Sort list of distances
        distances = sorted(distances.items(), key=lambda x: x[1])

        # Get n closest values
        closest = distances[0:n_closest]

        # Get filenames from ids
        array = []
        for item in closest:
             array.append(ids[item[0]])

        print('Cluster: {0} - Medoid: {1} -> {2}'.format(idx, ids[medoid_id], array))


def closest_docs2(x_train, id_medoids, labels, n_closest):
    # Read ids of files
    f = open('../documents/ids', 'r')
    ids = [line.rstrip('\n') for line in f]
    f.close()

    for idx in range(len(id_medoids)):
        # Get medoid ID
        medoid_id = id_medoids[idx]

        # Get train id from elements from cluster
        cluster_ids = list(np.where(labels == idx)[0])

        # Remove medoid from list of ids
        if len(cluster_ids) > 1:
            cluster_ids.remove(medoid_id)

        # Get all data using cluster ids - medoid_id
        X = x_train[cluster_ids]

        # Create a matrix of medoids to make distance measurement
        medoids = np.repeat(x_train[medoid_id], len(X)).reshape(X.shape)

        # Get distance from medoid matrix and data
        distances = paired_distances(medoids, X, 'euclidean')

        # Create indexes for the list of distances
        distances = {v: k for v, k in enumerate(distances)}

        # Sort list of distances
        distances = sorted(distances.items(), key=lambda x: x[1])

        # Get n closest values
        closest = distances[0:n_closest]

        # Get filenames from ids
        array = []
        for item in closest:
             array.append(ids[item[0]])

        print('Cluster: {0} - Medoid: {1} -> {2}'.format(idx, ids[medoid_id], array))
