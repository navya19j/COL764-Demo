from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
import sys
import os
import numpy as np
import json
import random
import re

def loadScores(
    result_file
):  
    scores = {}
    with open(result_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            words = re.split(" |\t", line)
            query_id = words[0]
            doc_id = words[2]
            if not query_id in scores:
                scores[query_id] = []
            scores[query_id].append(doc_id)
    return scores

def getRanklist(
    dataset, algorithm, embedding
): 
    offset = 0
    if dataset == "ap" and algorithm == "kmeans" and embedding == "bert":
        result_file = "K-Means_Bert_AP.txt"
    elif dataset == "ap" and algorithm == "kmeans" and embedding == "doc2vec":
        result_file = "K-Means_Doc2Vec_AP.txt"
    elif dataset == "ap" and algorithm == "clarans" and embedding == "bert":
        result_file = "Clarans_Bert_AP.txt"
    elif dataset == "ap" and algorithm == "clarans" and embedding == "doc2vec":
        result_file = "Clarans_Doc2Vec_AP.txt"
    elif dataset == "ziff" and algorithm == "kmeans" and embedding == "bert":
        result_file = "K-Means_Bert_ZF.txt"
    elif dataset == "ziff" and algorithm == "kmeans" and embedding == "doc2vec":
        result_file = "K-Means_Doc2Vec_ZF.txt"
    elif dataset == "ziff" and algorithm == "clarans" and embedding == "bert":
        result_file = "Clarans_Bert_ZF.txt"
    elif dataset == "ziff" and algorithm == "clarans" and embedding == "doc2vec":
        result_file = "Clarans_Doc2Vec_ZF.txt"
    else:
        if algorithm == "bm25":
            result_file = f"bm25-ranking-{dataset}"
        else:
            result_file = f"{embedding}-ranking-{dataset}"
        offset = 1

    query_map = {}
    with open(f"/home/chathur/Desktop/courses/COL764/project/COL764-Demo/code/rankings/{result_file}","r") as o:
        lines = o.readlines()
        text = " ".join(lines)
        text = text.split("\n")
        
        for line in text:
            words = line.split()
            if len(words) >= 3:
                qno = words[0]
                doc = words[1 + offset]
                score = words[2 + offset]
                if qno in query_map:
                    query_map[qno].append(doc)
                else:
                    query_map[qno] = [doc]         
    return query_map

def GetQueryNumber(text):

    tokens = text.split("Number:")
    for word in tokens:
        word = word.strip()
        if (len(word) > 0):
            return word
        
def getQueryTitles():

    queries = {}
    query_file = f"/home/chathur/Desktop/courses/COL764/project/COL764-Demo/code/queries/topics/trec12/topics.51-100.doc"

    with open(query_file, "r", errors="ignore") as f:
        content = f.read()

        bs_content = bs(content, 'lxml')
        all_queries = bs_content.find_all('top')

        for query in all_queries:

            qno = query.find("num").get_text()
            qno = GetQueryNumber(qno)
            content = query.find("title")
            text = content.get_text()[8:]
            queries[qno] = text

    return queries

def getDocumentContents(dataset):
    """ Returns a dictionary of {doc_name ---> doc_text}
    """

    count = 0

    docContent = {}
    required_tags = ["text"]
    all_dir = tqdm(os.listdir(f"/home/chathur/Desktop/courses/COL764/project/COL764-Demo/code/tipster_comp/{dataset}"))
    for num in all_dir:
        count += 1
        filepath = f"/home/chathur/Desktop/courses/COL764/project/COL764-Demo/code/tipster_comp/{dataset}/{num}"
        with open(filepath,"r",errors="ignore") as f:
            content = f.read()   
            bs_content = bs(content,'html.parser')
            all_doc = bs_content.find_all('doc')
            for doc in all_doc:
                docno = doc.find("docno").get_text().strip()
                tokentext = ""
                for tags in required_tags:
                    if tags == "head":
                        tokentext += "HEAD: "
                    text_content = doc.find_all(str(tags))
                    for text in text_content:
                        doc_content = text.get_text()
                        tokentext+=doc_content
                    if tags == "head":
                        tokentext += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                docContent[docno] = tokentext
#         if count > 1:  # WARNING: REMOVE!
#             break  
#         count += 1
    return docContent