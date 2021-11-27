import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
import numpy as np

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename) 
    return mat

def Kmeans(mat):
    
    distance = []
    clusters = []
    for cluster in range (10,500,20):
        model = KMeans(n_clusters=cluster,init='k-means++')
        model = model.fit(mat)
        distance.append(model.inertia_)
        print(cluster)
        clusters.append(cluster)
    
    print(clusters)
    print(distance)
    plt.plot(clusters, distance)
    plt.xlabel('k')
    plt.ylabel('Model Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def GetModel(mat,k):

    chosen_cluster = k
    print("Fitting Model")
    model = KMeans(n_clusters=chosen_cluster)
    model.fit(mat)
    print("Calculating Labels")
    labels = model.labels_
    print("Calculating Centers")
    centers = model.cluster_centers_
    
    return labels,centers

def silhoutte_score(mat,labels):

    return metrics.silhouette_score(mat, labels, metric='euclidean')

def Calinski_Harabasz(mat,labels):

    return metrics.calinski_harabasz_score(mat, labels)

def Davies_Bouldin_Index(mat,labels):

    return davies_bouldin_score(mat, labels)

if __name__ == "__main__":

    mat = GetMatrixFromDisk("mat.txt")
    # num_clusters = 2 #set number of clusters
    Kmeans(mat)
    # clusters,centers = GetModel(mat,num_clusters)
    # print(clusters)
    # print(centers)