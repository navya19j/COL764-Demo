from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
from sklearn import datasets
import numpy as np

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename) 
    return mat

def ConvToLists(mat):

    return mat.tolist()

def GetClusters(mat,num_clusters):

    iterations = 2 #set number of iterations
    neighbors = 5 #set number of neighbors to be compared
    clarans_instance = clarans(mat, num_clusters, iterations, neighbors)
    (ticks, result) = timedcall(clarans_instance.process)
    clusters = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()

    return clusters,medoids

if __name__ == "__main__":
    mat = GetMatrixFromDisk("docmatrix.txt")
    num_clusters = 5 #set number of clusters
    
    clusters,medoids = GetClusters(mat,num_clusters)