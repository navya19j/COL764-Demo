import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename) 
    return mat

def ConvTSV(mat):

    with open("centermat1.tsv","w") as o:

        for i in range (0,len(mat)):
            vec = mat[i]
            for term in vec:
                o.write(str(term))
                o.write("\t")
            o.write("\n")
    


# Use http://projector.tensorflow.org/ for visualization of tsv matrices

def add_value_labels(ax, spacing=5):

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.4f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)  

def plot_MAP_AP(yvals,titlee):


    
    xvals = ["K-Means","CLARANS","DB-SCAN","BM25"]
    

    # fig = plt.figure(figsize = (5, 3))

    # plt.bar(xvals, bert, color ='gray',
    #     width = 0.3)

    freq_series = pd.Series(yvals)
    my_colors = list(['b', 'r', 'g', 'y', 'k'])
 
    # plt.xlabel("Retrieval Method")
    # plt.ylabel("mAP Score")
    # plt.title("Clustering with BERT embedding v/s BM25")
    # plt.show()

    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=9)  # fontsize of the figure title

    plt.figure(figsize=(8, 5))
    ax = freq_series.plot(kind="bar",color =my_colors)
    ax.set_title(titlee)
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("mAP Score")
    ax.set_xticklabels(xvals,rotation=45, ha='left')

    rects = ax.patches
    

    # Make some labels.
    # labels = [f"label{i}" for i in range(len(rects))]

    add_value_labels(ax, spacing=5)

    plt.show()

    # fig = plt.figure(figsize = (10, 5))

    # plt.bar(xvals, doc2vec, color ='blue',
    #     width = 0.4)
 
    # plt.xlabel("Retrieval Method")
    # plt.ylabel("mAP Score")
    # plt.title("Clustering with Doc2Vec embedding v/s BM25")
    # plt.show()

def plot():

    bm25ap = [0.3440145458986036, 0.3310199385712331, 0.31785408974893337, 0.2690420700911771, 0.26265440022118725, 0.24230070776584556, 0.2325012840070273, 0.20026029666783743, 0.18632996490307988, 0.15566280194523863, 0.14789720998055197, 0.14003711909512323, 0.1320983667005804, 0.11499309038210319, 0.11323153959689246][0:10]
    bm25zif = [1.0, 1.0, 1.0, 1.0, 0.9602951503222242, 0.8611111111111112, 0.848421322171848, 0.8301254723398109, 0.7877252965567625, 0.6896144420360651, 0.6742578396649442, 0.6610578852467778, 0.43724982746721874, 0.4286331604642242, 0.3669716146234724][0:10]

    kmeansapdocvec = sorted([0.04, 0.07893410359073144, 0.015151515151515152, 0.011904761904761904, 0.07226087624070925, 0.010446437370651424, 0.31415850064978523, 0.007692307692307693, 0.1404235171067261, 0.022823197651572938, 0.022066942719116633, 0.12049696626902509, 0.06666666666666667, 0.010169654906497011, 0.036713106295149636, 0.28562062238250085, 0.006493506493506494, 0.21823752055279147, 0.041666666666666664, 0.0136986301369863],reverse=True)[0:10]
    kmeansapbert = sorted([0.010526315789473684, 0.01, 0.027777777777777776, 0.3333333333333333, 0.038461538461538464, 0.3366013071895425, 1.0, 0.008928571428571428, 0.06086956521739131, 0.09964157706093191, 0.022222222222222223, 0.016129032258064516, 0.9206826273715201, 0.125, 0.019230769230769232, 0.5, 0.39285714285714285, 0.6626984126984126, 0.5588235294117647, 1.0],reverse=True)[0:10]

    kmeanszifdocvec = sorted([0.023809523809523808, 0.12618344793505346, 0.08014592578229027, 0.1582610297515496, 0.10857341558317715, 0.013333333333333334, 0.11848192714000957, 0.014705882352941176, 0.011765095777821966, 0.17202184169207985, 0.07635776469239727, 0.07243115145794088, 0.133557858589153, 0.17080123175319215],reverse=True)[0:10]
    kmeanszifbert = sorted([0.020833333333333332, 0.40966750823825315, 0.24584448328364322, 0.2571323474085066, 0.09923416036728186, 0.11374220250312239, 0.00980392156862745, 0.28448275862068967, 0.038810277616247765, 0.05954106280193236, 0.5543352304072432, 0.08568554468526428, 0.22713118530424525],reverse = True)[0:10]

    x = [1,2,3,4,5,6,7,8,9,10]

    plt.plot(x,bm25ap, marker='x',color='blue',label='BM25')
    plt.plot(x,kmeansapdocvec, marker='x',color='red',label='K-Means with Doc2Vec Embedding')
    plt.plot(x,kmeansapbert, marker='x',color='green',label='K-Means with BERT Embedding')
    
    plt.xlabel("Retrieval Model")
    plt.ylabel("AP Score")
    plt.title("Top 10 AP Scores - AP Dataset")
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    # To load the display window
    plt.show()

    plt.plot(x,bm25zif, marker='x',color='blue',label='BM25')
    plt.plot(x,kmeanszifdocvec, marker='x',color='red',label='K-Means with Doc2Vec Embedding')
    plt.plot(x,kmeanszifbert, marker='x',color='green',label='K-Means with BERT Embedding')
    
    plt.xlabel("Retrieval Model")
    plt.ylabel("AP Score")
    plt.title("Top 10 AP Scores - ZF Dataset")
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    # To load the display window
    plt.show()

if __name__ == "__main__":

    doc2vec_zf = [0.033491359,0.046170034,0.0405,0.242088356]
    title = "Clustering with Doc2Vec embedding v/s BM25 - ZF Dataset"
    plot_MAP_AP(doc2vec_zf,title)

    doc2vec_ap = [0.033330762,0.041076875 ,0.064399,0.091872847]
    title = "Clustering with Doc2Vec embedding v/s BM25 - AP Dataset"
    plot_MAP_AP(doc2vec_ap,title)

    bert_ap = [0.119876425,0.191080783,0.352482576,0.091872847]
    title = "Clustering with BERT embedding v/s BM25 - AP Dataset"
    plot_MAP_AP(bert_ap,title)

    bert_zf = [0.078175149,0.047689964,0.1184,0.242088356]
    title = "Clustering with BERT embedding v/s BM25 - ZF Dataset"
    plot_MAP_AP(bert_zf,title)

    mat = GetMatrixFromDisk("centers.txt")
    ConvTSV(mat)

    plot()


