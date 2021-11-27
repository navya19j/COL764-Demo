from pyclustering.cluster.clarans import clarans;
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.utils import timedcall;
import numpy as np
from pyclustering.cluster.silhouette import silhouette

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename) 
    return mat

def ConvToLists(mat):

    return mat.tolist()

def getsilhouette(mat,clusters):

    return silhouette(mat, clusters).process().get_score()

def GetClusters(mat,num_clusters):

# max_neighbors => p% of (k*(n-k))

    iterations = 2 #set number of iterations
    neighbors = 250 #set number of neighbors to be compared
    clarans_instance = clarans(mat, num_clusters, iterations, neighbors)
    (ticks, result) = timedcall(clarans_instance.process)

    print(ticks)

    clusters = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()

    # vis = cluster_visualizer_multidim()
    # vis.append_clusters(clusters,mat.tolist(),marker="*",markersize=5)
    # vis.show(pair_filter=[[0, 1], [0, 2]])

    return clusters,medoids


# if __name__ == "__main__":

#     mat = GetMatrixFromDisk("docmatrix.txt")
#     num_clusters = 5 #set number of clusters
    
#     clusters,medoids = GetClusters(mat,num_clusters)