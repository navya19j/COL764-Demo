"""BERT For Document Similarity Computation

Running this script comptues the BERT embeddings
of a set of documents and stores it in some file for later use.

To generate and store the embeddings for the AP dataset, run the following line on the terminal

$ python3 bert.py 2 embeddings/bert-embedding embeddings/bert-docmap
"""

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np
import pandas as pd
import json
import random
import re

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("logs/agreement_all.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def GetDocumentList(CollectionName):
    """ Returns a dictionary of {doc_name ---> doc_text}
    """

    count = 0

    docContent = {}
    required_tags = ["text"]
    all_dir = tqdm(os.listdir(os.path.join("tipster_comp",CollectionName)),position=0, leave=True)
    for num in all_dir:
        count += 1
        filepath = os.path.join(os.path.join("tipster_comp",CollectionName),num)
        with open(filepath,"r",errors="ignore") as f:
            content = f.read()   
            bs_content = bs(content,'html.parser')
            all_doc = bs_content.find_all('doc')
            for doc in all_doc:
                docno = doc.find("docno").get_text().strip()
                for tags in required_tags:
                    text_content = doc.find_all(str(tags))
                    tokentext = ""
                    for text in text_content:
                        doc_content = text.get_text()
                        tokentext+=doc_content
                    docContent[docno] = tokentext
        # if count > 5:  # WARNING: REMOVE!
        #     break  

    return docContent


def GetQueryNumber(text):

    tokens = text.split("Number:")
    for word in tokens:
        word = word.strip()
        if (len(word) > 0):
            print(word)
            return word

def GetQueryList():

    queries = {}
    query_file = os.path.join("queries", "topics", "trec12", "topics.51-100.doc")

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


def getModel(modelname):
    """ Returns the BERT model
    """
    model = SentenceTransformer(modelname)
    return model

def ExtractQrels(filename):
    """ Extracts qrels from the given file
    """
    qrels = []
    with open(filename,"r") as o:
        lines = o.readlines()
        text = " ".join(lines)
        text = text.replace("\n"," ")
        text = text.replace("\\"," ")
        text = re.split(" ",text)

    for word in text:
        if "AP" in word:
            qrels.append(word)
            # print(word)

    return qrels

def getDocMatrix(data, modelname):
    """ Returns BERT embeddings of the given documents using the given model
    
    Also returns a mapping from index # in the embeddings matrix to doc_name 
    """
    model = getModel(modelname)
    DocMatrix = []
    i = 0
    DocMap = {}

    for doc in tqdm(data):
        text = data[doc]
        vec = model.encode([text])[0] 
        DocMatrix.append(vec)
        DocMap[i] = doc 
        i += 1
    
    DocMatrix = np.array(DocMatrix)
    return DocMatrix,DocMap

def RandomSampleTest(qrels,data):
    """ Randomly sample a subset of documents from data.

    Number of sampled documents is given by the parameter thresh.
    All documents in the qrels file are also included, so sampled documents will likely exceed
    the given threshold
    """
    # thresh = 500  # WARNING: CHANGE BACK TO 8000!
    test_data = dict(random.sample(data.items(), thresh))
    for doc in qrels:
        if doc not in test_data and doc in data:
            test_data[doc] = data[doc]
    return test_data

if __name__ == "__main__":

    if sys.argv[1] == "1":
        CollectionName = "wsj"
    elif sys.argv[1] == "2":
        CollectionName = "ap"
    else:
        CollectionName = "ziff"

    modelname = "bert-base-nli-mean-tokens"
    
    logger.info(f"STARTED READING DOCUMENTS...")
    data = GetDocumentList(CollectionName)  
    logger.info(f"NUMBER OF DOCUMENTS = {len(data)}")

    # logger.info(f"EXTRACTING QRELS...")
    # qrels = ExtractQrels("qrels.rtf")

    # logger.info(f"SAMPLING SUBSET OF DOCUMENTS...")
    # data_subset = RandomSampleTest(qrels,data)
    # logger.info(f"NUMBER OF SAMPLED DOCUMENTS = {len(data_subset)}")

    # logger.info(f"FINDING THE BERT DOCUMENT REPRESENTATION OF SAMPLED DOCUMENTS, MODEL = {modelname}")
    # DocMatrix, DocMap = getDocMatrix(data_subset, modelname)

    # logger.info(f"SAVING DOC MATRIX...")
    # np.savetxt(f"embeddings/bert-embedding-{CollectionName}", DocMatrix)

    # logger.info(f"SAVING DOCUMENT NUMBER ---> DOCUMENT NAME MAPPING...")
    # with open(f"embeddings/bert-docmap-{CollectionName}","w") as o:
    #     o.write(json.dumps(DocMap))



    logger.info(f"STARTED READING QUERIES...")
    data = GetQueryList()
    logger.info(f"NUMBER OF QUERIES = {len(data)}")

    QueryMatrix, QueryMap = getDocMatrix(data, modelname)
    logger.info(f"SAVING QUERY MATRIX...")
    np.savetxt(f"embeddings/bert-query-embedding-{CollectionName}", QueryMatrix)

    logger.info(f"SAVING QUERY NUMBER ---> QUERY NAME MAPPING...")
    with open(f"embeddings/bert-querymap-{CollectionName}", "w") as o:
        o.write(json.dumps(QueryMap))

    
