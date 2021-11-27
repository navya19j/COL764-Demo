from tqdm import tqdm
from bs4 import BeautifulSoup as bs
import sys
import os
import time
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from clarans import *
from kmeans import *
from scipy import spatial
from docVec import *

def GetQueryNumber(text):

    tokens = text.split("Number:")
    for word in tokens:
        word = word.strip()
        if (len(word)>0):
            return word

def CosineSim(vector1,vector2):
    cosine_similarity = 1 - spatial.distance.cosine(np.array(vector1,dtype=np.float32), np.array(vector2,dtype=np.float32))

    return abs(cosine_similarity)

def GetQueriesBert(filename):

    queries = {}
    query_file = os.path.join(os.getcwd(),"queries","topics","trec12",filename)

    with open(query_file,"r",errors="ignore") as f:
        content = f.read()

        bs_content = bs(content,'html.parser')
        all_queries = bs_content.find_all('top')

        for query in all_queries:

            qno = query.find("num").get_text()
            qno = GetQueryNumber(qno)
            content = query.find("title")
            
            text = content.get_text()
            queries[qno] = text
            print(qno)

    return queries

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

def GetQueries(filename):

    queries = {}
    query_file = os.path.join(os.getcwd(),"queries","topics","trec12",filename)
    loop = tqdm(total=50)
    with open(query_file,"r",errors="ignore") as f:
        content = f.read()

        bs_content = bs(content,'html.parser')
        all_queries = bs_content.find_all('top')

        for query in all_queries:

            qno = query.find("num").get_text()
            qno = GetQueryNumber(qno)

            content = query.find("title")
            
            text = content.get_text()

            queries[qno] = text
            loop.update(1)

    return queries

def getQueryVector(query,model):
    vec = getVector(query,model)
    return vec

def getQueryDict(queries,model):
    vect = {}
    loop = tqdm(total=50)
    for query in queries:
        tokens = GetTokens(queries[query])
        vect[query] = getQueryVector(tokens,model)
        # print(vect[query].shape)
        loop.update(1)

    return vect

def getQueryVectorBert(query,model):

    vec = model.encode([query])[0]
    return vec

def getQueryDictBert(queries):

    modelname = "bert-base-nli-mean-tokens"
    model = SentenceTransformer(modelname)
    vect = {}
    loop = tqdm(total=50)
    for query in queries:
        print(query)
        vect[query] = getQueryVectorBert(queries[query],model)
        # print(vect[query].shape)
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

def AssignQueries(query_dict,centers,mat,cluster_type):

    rank = {}
    for query in query_dict:
        # print("Ranking to query: " + query)
        rank[query] = MostSimilarCluster(query,centers,mat,cluster_type)
        # print(rank[query])

    return rank

def LabelDict(labels,docmap,cluster_type):

    labeldict = {}
    if (cluster_type=="kmeans"):
        for i in range (0,len(labels)):
            if labels[i] in labeldict:
                labeldict[labels[i]].append(docmap[i])
            else:
                labeldict[labels[i]] = [docmap[i]]
    else:
        for i in range (0,len(labels)):
            for j in labels[i]:
                if i in labeldict:
                    labeldict[i].append(docmap[j])
                else:
                    labeldict[i] = [docmap[j]]

    return labeldict

def readDocMap(filename):

    with open(filename) as f:
        data = f.read()
        js = json.loads(data)
        jslist = list(js.keys())
    # print("Doc Map List")
    # print(jslist)

    return jslist,js

def RankDocs(labeldict,assigned,docmapdict,query_dict,mat):

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

def OutputRankings(ranklist,rankfile):
    
    with open(rankfile,"w") as o:
        for query in ranklist:
            i = 0
            for doc in ranklist[query]:
                sim_score = ranklist[query][doc]
                o.write(query+" "+doc+" "+str(sim_score)+"\n")
                i+=1
                if (i>=99):
                    break
                    

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

# def OutputQueryDict(qudict):

#     with open("queryDict.txt","w") as o:
#         o.write(json.dumps(qudict))

def OutputQueryDict(qudict):

    mat = []
    map = {}
    i = 0

    for query in qudict:
        mat.append(qudict[query])
        map[query] = i
        i+=1

    np.savetxt("ZF_mat.txt",np.array(mat))

    with open("ZF_map.txt","w") as o:
        o.write(json.dumps(map))

def get_results(embedding,cluster_type,matname,dataset,modelname,map):

    mat = GetMatrixFromDisk(matname)
    print(mat.shape)
    if (embedding == "doc2vec"):

        queries = GetQueries("topics.51-100.doc")
        query_dict = getQueryDict(queries,modelname)

        # OutputQueryDict(query_dict)
        if (cluster_type == "kmeans"):
            num_clusters = 20 #set number of clusters
            print("Calculating K-Means -20")
            clusters,centers = GetModel(mat,num_clusters)
            outputClusters(clusters,centers)
            print("silhoutte score: ")
            print(silhoutte_score(mat,clusters))
            print("Calini-Harasbasz: ")
            print(Calinski_Harabasz(mat,clusters))
            print("Davies Bouldin: ")
            print(Davies_Bouldin_Index(mat,clusters))

            assignedClusters = AssignQueries(query_dict,centers,mat,cluster_type)
            docmap,docmapdict = readDocMap(map)
            dictLabel = LabelDict(clusters,docmap,cluster_type)
            rankfile = "K-Means_Doc2Vec_" + dataset + ".txt"
            ranklist = RankDocs(dictLabel,assignedClusters,docmapdict,query_dict,mat)

            OutputRankings(ranklist,rankfile)
            
        elif (cluster_type == "clarans"):
            num_clusters = 20 #set number of clusters
            start = time.time()
            print("Calculating Clarans")
            clusters,centers = GetClusters(mat,num_clusters)
            
            print("Done")
            end = time.time()
            print(end-start)

            outputClustersClarans(clusters,centers)
            print(mat.shape)
            assignedClusters = AssignQueries(query_dict,centers,mat,cluster_type)
            docmap,docmapdict = readDocMap(map)
            dictLabel = LabelDict(clusters,docmap,cluster_type)
            ranklist = RankDocs(dictLabel,assignedClusters,docmapdict,query_dict,mat)
            rankfile = "Clarans_Doc2Vec_" + dataset + ".txt"

            OutputRankings(ranklist,rankfile)

            # print("silhoutte score: ")
            # print(getsilhouette(mat,clusters))

    elif (embedding=="bert"):

        queries = GetQueriesBert("topics.51-100.doc")
        query_dict = getQueryDictBert(queries)

        if (cluster_type == "kmeans"):
            num_clusters = 20 #set number of clusters
            print("Calculating K-Means -20")
            clusters,centers = GetModel(mat,num_clusters)
            outputClusters(clusters,centers)
            print("silhoutte score: ")
            print(silhoutte_score(mat,clusters))
            print("Calini-Harasbasz: ")
            print(Calinski_Harabasz(mat,clusters))
            print("Davies Bouldin: ")
            print(Davies_Bouldin_Index(mat,clusters))

            assignedClusters = AssignQueries(query_dict,centers,mat,cluster_type)
            docmap,docmapdict = readDocMap(map)
            dictLabel = LabelDict(clusters,docmap,cluster_type)
            ranklist = RankDocs(dictLabel,assignedClusters,docmapdict,query_dict,mat)
            rankfile = "K-Means_Bert_" + dataset + ".txt"
            OutputRankings(ranklist,rankfile)
            
        elif (cluster_type == "clarans"):
            num_clusters = 20 #set number of clusters
            start = time.time()
            print("Calculating Clarans")
            clusters,centers = GetClusters(mat,num_clusters)
            
            print("Done")
            end = time.time()
            print(end-start)

            print(mat.shape)
            assignedClusters = AssignQueries(query_dict,centers,mat,cluster_type)
            docmap,docmapdict = readDocMap(map)
            dictLabel = LabelDict(clusters,docmap,cluster_type)
            ranklist = RankDocs(dictLabel,assignedClusters,docmapdict,query_dict,mat)

            rankfile = "Clarans_Bert_" + dataset + ".txt"
            OutputRankings(ranklist,rankfile)

if __name__ == "__main__":

    stop_words = GetStopwords("stopwords.txt")

    matname = "Bert_ZF_mat.txt"
    dataset = "ZF"
    embedding = "bert"
    cluster_type = "clarans"
    modelname = "d2vap.model"
    map = "Bert_ZF_map.txt"
    get_results(embedding,cluster_type,matname,dataset,modelname,map)
    print("Done"+matname)

    matname = "Bert_AP_mat.txt"
    dataset = "AP"
    embedding = "bert"
    cluster_type = "clarans"
    modelname = "d2vap.model"
    map = "Bert_AP_map.txt"
    get_results(embedding,cluster_type,matname,dataset,modelname,map)
    print("Done"+matname)
    
    matname = "doc2vec_ZF_mat.txt"
    dataset = "ZF"
    embedding = "doc2vec"
    cluster_type = "clarans"
    modelname = "d2vzf.model"
    map = "doc2vec_ZF_map.txt"
    get_results(embedding,cluster_type,matname,dataset,modelname,map)
    print("Done"+matname)
    
    matname = "doc2vec_AP_mat.txt"
    dataset = "AP"
    embedding = "doc2vec"
    cluster_type = "clarans"
    modelname = "d2vap.model"
    map = "doc2vec_AP_map.txt"
    get_results(embedding,cluster_type,matname,dataset,modelname,map)
    print("Done"+matname)
