from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.join(sys.path[0], "algorithms"))
sys.path.append(os.path.join(sys.path[0], "representations"))
import numpy as np
import json
import re
import doc2vec
from doc2vec import *
from kmeans import *
from clarans import *
from scipy import spatial

def GetQueryNumber(text):

    tokens = text.split("Number:")
    for word in tokens:
        word = word.strip()
        if (len(word)>0):
            return word

def CosineSim(vector1,vector2):
    cosine_similarity = 1 - spatial.distance.cosine(np.array(vector1,dtype=np.float32), np.array(vector2,dtype=np.float32))

    return abs(cosine_similarity)

def GetTokens(text_content):
    
    ans = []
    # p = PorterStemmer()
    text = text_content.replace("-",":")
    text_content = text.replace("\n",":")
    text_content = text.replace("[",":")
    text_content = text.replace("]",":")
    tokenized_content = re.split(r'''[ ',.=:(_);{}?`"\n]''',text_content)
    for token in tokenized_content:
        token = token.lower()
        if (len(token)>2):
            if (token not in stop_words and not re.search('[0-9]+',token)):
                # token = p.stem(token,0,len(token)-1)
                ans.append(token)
                    
    return ans

def GetStopwords(file):

    stopwordFile = open(os.path.join("tipster_comp",file))
    stop_words = set()
    stop_words.add('&')
    stop_words.add('=')
    stop_words.add('"')
    stop_words.add("topic")
    stop_words.add('<')
    stop_words.add('>')
    lines = stopwordFile.readlines()
    words = " ".join(lines)
    words_list = re.split('''[\n]''',words)

    for word in words_list:
        stop_words.add(word)

    return stop_words

def GetMatrixFromDisk(filename):

    mat = np.loadtxt(filename)
    return mat

def GetQueries(filename):

    queries = {}
    query_file = os.path.join(os.getcwd(),"queries","topics","trec12",filename)

    with open(query_file,"r",errors="ignore") as f:
        content = f.read()

        bs_content = bs(content,'lxml')
        all_queries = bs_content.find_all('top')

        for query in all_queries:

            qno = query.find("num").get_text()
            qno = GetQueryNumber(qno)

            content = query.find("title")
            
            text = content.get_text()[8:]
            cont = GetTokens(text)

            queries[qno] = cont

    return queries

def getQueryVector(query,model, f):
    vec = f(query, model)
    # vec = doc2vec.getVector(query,model)
    return vec

def getQueryDict(queries,model):
    vect = {}
    loop = tqdm(total=50)
    for query in queries:
        vect[query] = getQueryVector(queries[query],model)
        print(vect[query].shape)
        loop.update(1)

    return vect

def MostSimilarCluster(queryVect,centers,mat,model):

    sim = {}
    if (model == "kmeans"):
        for i in range (0,len(centers)):
            labVec = centers[i]
            # print(labVec)
            val = CosineSim(labVec,queryVect)
            # print("Cosine sim")
            # print(i)
            # print(val)
            sim[i] = val
    else:
        for i in range (0,len(centers)):
            labVec = mat[centers[i]]
            # print(labVec)
            val = CosineSim(labVec,queryVect)
            # print("Cosine sim")
            # print(i)
            # print(val)
            sim[i] = val


    sim = dict(sorted(sim.items(), key=lambda item: item[1]))
    first = list(sim.keys())[-1]
    # print(first)

    return first

def AssignQueries(query_dict,centers,mat):

    rank = {}
    for query in query_dict:
        # print("Ranking to query: " + query)
        rank[query] = MostSimilarCluster(query,centers,mat,sys.argv[1])
        # print(rank[query])

    return rank

def LabelDict(labels,docmap):

    labeldict = {}
    if (sys.argv[1]=="kmeans"):
        for i in range (0,len(labels)):
            if labels[i] in labeldict:
                labeldict[labels[i]].append(docmap[i])
            else:
                labeldict[labels[i]] = [docmap[i]]
    else:
        for i in range (0,len(labels)):
            if i in labeldict:
                labeldict[i].append(docmap[i])
            else:
                labeldict[i] = [docmap[i]]
    # print("Label to Dict")

    return labeldict

def readDocMap(filename):

    with open(filename) as f:
        data = f.read()
        js = json.loads(data)
        jslist = list(js.keys())
    # print("Doc Map List")
    # print(jslist)

    return jslist,js

def RankDocs(labeldict,assigned):

    ranklist = {}
    for query in assigned:
        rank = {}
        for doc in labeldict[assigned[query]]:
            index = int(docmapdict[doc])
            docVec = mat[index]
            queryVec = query_dict[query]
            rank[doc] = CosineSim(docVec,queryVec)
        sortrank = dict(sorted(rank.items(), key=lambda item: item[1],reverse=True))
        ranklist[query] = sortrank
    return ranklist

def OutputRankings(ranklist):

    with open("result.txt","w") as o:
        for query in ranklist:
            for doc in ranklist[query]:
                sim_score = ranklist[query][doc]
                o.write(query+" "+doc+" "+str(sim_score)+"\n")

def outputClusters(clusters,centers):

    centers = np.array(centers)
    clusters = np.array(clusters)
    np.savetxt("centers.txt",centers)
    np.savetxt("clusters.txt",clusters)

def outputClustersClarans(clusters,centers):

    centers_out = []
    for term in centers:
        centers_out.append(int(term))
    clusters_out = []
    for word in clusters:
        temp = []
        for x in word:
            temp.append(int(x))
        clusters_out.append(np.array(temp))

    np.savetxt("centers.txt",np.array(centers_out))
    np.savetxt("clusters.txt",np.array(clusters_out), fmt='%s')

def OutputQueryDict(qudict):

    with open("queryDict.txt","w") as o:
        o.write(json.dumps(qudict))


if __name__ == "__main__":
    stop_words = GetStopwords("stopwords.txt")
    queries = GetQueries("topics.51-100.doc")
    # print("Query Dict")

    # print(query_dict)
    mat = GetMatrixFromDisk("embeddings/bert-embedding-ap")
    print(mat.shape)

    if (sys.argv[1] == "kmeans"):
        num_clusters = 50 #set number of clusters
        print("Calculating K-Means")
        clusters,centers = GetModel(mat,num_clusters)
        outputClusters(clusters,centers)
        assignedClusters = AssignQueries(query_dict,centers,mat)
        docmap,docmapdict = readDocMap("map.txt")
        dictLabel = LabelDict(clusters,docmap)
        ranklist = RankDocs(dictLabel,assignedClusters)
        OutputRankings(ranklist)

    elif (sys.argv[1] == "clarans"):
        modelname = "d2v.model"
        query_dict = getQueryDict(queries, modelname)
        num_clusters = 50 #set number of clusters
        print("Calculating Clarans")
        clusters,centers = GetClusters(mat,num_clusters)

        outputClustersClarans(clusters,centers)

        assignedClusters = AssignQueries(query_dict,centers,mat)
        docmap,docmapdict = readDocMap("map.txt")
        dictLabel = LabelDict(clusters,docmap)
        ranklist = RankDocs(dictLabel,assignedClusters)
        OutputRankings(ranklist)
    #let labels be vectors of chosen cluster centers

