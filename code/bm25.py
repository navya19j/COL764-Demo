from rank_bm25 import BM25Okapi
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
    all_dir = tqdm(os.listdir(os.path.join(
        "tipster_comp", CollectionName)), position=0, leave=True)
    for num in all_dir:
        filepath = os.path.join(os.path.join(
            "tipster_comp", CollectionName), num)
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
            bs_content = bs(content, 'html.parser')
            all_doc = bs_content.find_all('doc')
            for doc in all_doc:
                docno = doc.find("docno").get_text().strip()
                for tags in required_tags:
                    text_content = doc.find_all(str(tags))
                    tokentext = ""
                    for text in text_content:
                        doc_content = text.get_text()
                        tokentext += doc_content
                    docContent[docno] = tokentext
        if count == 1:  # WARNING: REMOVE!
            break
        count += 1

    return docContent


def GetQueryNumber(text):

    tokens = text.split("Number:")
    for word in tokens:
        word = word.strip()
        if (len(word) > 0):
            return word


def GetQueryList():

    queries = {}
    query_file = os.path.join(
        "queries", "topics", "trec12", "topics.51-100.doc")

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
    
if __name__ == "__main__":
    
    CollectionName = sys.argv[1]

    logger.info(f"STARTED READING DOCUMENTS...")
    data = GetDocumentList(CollectionName)
    logger.info(f"NUMBER OF DOCUMENTS = {len(data)}")

    logger.info(f"STARTED READING QUERIES...")
    query_data = GetQueryList()
    logger.info(f"NUMBER OF QUERIES = {len(data)}")

    with open(f"embeddings/doc2vec-docmap-{CollectionName}", "r") as f:
        query_map  = json.loads(f.read())
    
    relevant_files = set(query_map.values())
    reduced_data = {}
    for doc_id, text in data.items():
        if doc_id in relevant_files:
            reduced_data[doc_id] = text
    
    data = reduced_data
    logger.info(f"NUMBER OF DOCUMENTS BEING PROCESSED = {len(data)}")

    doc_ids = []
    corpus = []

    for doc_id, doc_text in data.items():
        doc_ids.append(doc_id)
        corpus.append(doc_text)
    
    tokenized_corpus = [doc.split(" ") for doc in corpus]
  
    bm25 = BM25Okapi(tokenized_corpus)
    for query_id, query_text in query_data.items():
        tokenized_query = query_text.split(" ")
        scores = bm25.get_scores(tokenized_query)
        sorted_scores = [(scores[i], doc_ids[i]) for i in range(len(scores))]
        sorted_scores.sort(reverse=True)
        # print(query_id)
        # print(query_text)
        # print(sorted_scores[:10])
        # for i in range(15):
        #     print(sorted_scores[i][0])
        with open(f"rankings/demo-bm25-ranking-{CollectionName}-double-check", "a") as f:
            for i in range(min(len(sorted_scores), 100)):
                f.write(f"{query_id} 0 {sorted_scores[i][1]} {sorted_scores[i][0]}\n")




    


