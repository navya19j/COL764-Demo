import numpy as np

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename) 
    return mat

def ConvTSV(mat):

    with open("centermat.tsv","w") as o:

        for i in range (0,len(mat)):
            vec = mat[i]
            for term in vec:
                o.write(str(term))
                o.write("\t")
            o.write("\n")
    
mat = GetMatrixFromDisk("centers.txt")
ConvTSV(mat)

# Use http://projector.tensorflow.org/ for visualization of tsv matrices
