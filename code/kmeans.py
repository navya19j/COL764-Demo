import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename)
    return mat


def Kmeans(mat):

    distance = []
    clusters = [500, 1000, 2000, 3000]
    for cluster in range(500, 3000, 500):
        model = KMeans(n_clusters=cluster)
        model = model.fit(mat)
        distance.append(model.inertia_)
        print(cluster)

    print(clusters)
    print(distance)
    plt.plot(clusters, distance)
    plt.xlabel('k')
    plt.ylabel('Model Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def GetModel(mat, k):

    chosen_cluster = k
    model = KMeans(n_clusters=chosen_cluster, init='k-means++')
    model.fit(mat)
    labels = model.labels_
    centers = model.cluster_centers_

    return labels, centers


if __name__ == "__main__":

    mat = GetMatrixFromDisk("mat.txt")
    num_clusters = 5  # set number of clusters
    Kmeans(mat)
    # clusters,centers = GetModel(mat,num_clusters)
