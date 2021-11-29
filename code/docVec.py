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

# Tokenisation Algorithm : Get Tokenised Content from Text corpus
def GetTokens(text_content):
    
    ans = []
    # p = PorterStemmer()
    text = text_content.replace("-",":")
    text_content = text.replace("\n",":")
    tokenized_content = re.split(r'''[ `',.=:(_);{}?`"\n]''',text_content)
    for token in tokenized_content:
        token = token.lower()
        if (len(token)>2):
            if (token not in stop_words and not re.search('[0-9]+',token)):
                # token = p.stem(token,0,len(token)-1)
                ans.append(token)
                    
    return ans

# Get Mapping of Doc Name to Text in the doc"
def GetDocumentList(CollectionName):
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
        if count == 1:  # WARNING: REMOVE!
            break  
        count += 1

    return docContent

# Get set of stopwords from nltk + given stopwords file
def GetStopwords(file):

    stopwordFile = open(os.path.join("tipster_comp",file))
    stop_words = set()
    stop_words.add('&')
    stop_words.add('=')
    stop_words.add('"')
    stop_words.add('<')
    stop_words.add('>')
    lines = stopwordFile.readlines()
    words = " ".join(lines)
    words_list = re.split('''[\n]''',words)

    for word in words_list:
        stop_words.add(word)

    return stop_words

# Train Doc2Vec model on documents in the list doc2vc
def TrainDocVec(tagset,filename):

    max_epochs = tqdm(total=1)
    vec_size = vector_length
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, window = 15,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)

    model.build_vocab(tagset)

    for epoch in range(1):
        model.train(tagset,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
        max_epochs.update(1)

    # print(model.most_similar("book"))
    model.save(filename)
    # print("Model Saved")

# Load Trained Model from Disk
def getModel(model):
    modelname = Doc2Vec.load(model)
    return modelname

# Get Vector Representation of a Bag of Words from model "modelname"
def getVector(words,modelname):

    model = getModel(modelname)
    v1 = model.infer_vector(words,epochs=1)
    return np.array(v1)

def TagAllDocs(data):

    tagSet = []
    tokenDict = {}
    j=0

    for i in tqdm(data):
        tag_val = i
        words = GetTokens(data[i])

        tagSet.append(TaggedDocument(words, tags=[str(tag_val)]))

    return tagSet

def PartitionDocs(data):

    content = pd.Series(data)
    train , test  = [i.to_dict() for i in train_test_split(content, train_size = 0.6)]

    return train,test

def TestModel(model,testdata):

    # print(model.epochs)

    for doc in testdata:
        text = testdata[doc]
        tokens = GetTokens(text)
        vec = getVector(tokens,model)

def ExtractQrels(filename):

    qrels = []
    with open(filename,"r") as o:
        lines = o.readlines()
        text = " ".join(lines)
        text = text.replace("\n"," ")
        text = text.replace("\\"," ")
        text = text.replace("}"," ")
        text = text.replace("{"," ")
        text = re.split(" ",text)

    for word in text:
        if "ZF" in word:
            qrels.append(word)
            # print(word)
    # print(qrels)
    return qrels

def getDocMatrix(data,modelname):

    model = getModel(modelname)
    DocMatrix = np.zeros((len(data),vector_length))
    i = 0
    DocMap = {}
    avglen = 0
    for doc in tqdm(data):
        text = data[doc]
        # print(text)
        tokens = GetTokens(text)
        avglen+=len(tokens)
        vec = model.dv[doc]
        DocMatrix[i] = vec
        DocMap[doc] = i
        i+=1

    # print("Average Length")
    # print(avglen/len(data))

    print(DocMatrix.shape)

    return DocMatrix,DocMap

def computeSimilarities(data,modelname):

    model = getModel(modelname)
    # print(model.wv.index_to_key)

    docmatrix = np.zeros((len(data),len(data)))
    i=0
    j=0
    loop = tqdm(len(docmatrix)**2)
    for doc1 in data:
        for doc2 in data:
            docmatrix[i][j] = model.dv.n_similarity(data[doc1],data[doc2])
            # print(docmatrix[i][j])
            loop.update(1)
            j+=1
        i+=1
    
    return docmatrix


def RandomSampleTest(qrels,data,thresh):
    thresh = 3
    test_data = dict(random.sample(data.items(), thresh))
    # print(test_data)
    for doc in qrels:
        if doc not in test_data and doc in data:
            test_data[doc] = data[doc]
    #         print(doc)
    # print("done")
    return test_data

if __name__ == "__main__":
    
    train = True

    if sys.argv[1] == "1":
        CollectionName = "wsj"
    elif sys.argv[1] == "2":
        CollectionName = "ap"
    else:
        CollectionName = "ziff"

    modelname = "d2v.model"
    
    vector_length = 1000
    data = GetDocumentList(CollectionName)
    # print(len(data))

    stop_words = GetStopwords("stopwords.txt")

    qrels = ExtractQrels("qrels_ZF.rtf")
    train_set = data

    # train_set = data

    if (train == True):
        TagData = TagAllDocs(train_set)
        TrainDocVec(TagData,modelname)

    DocMatrix,DocMap = getDocMatrix(RandomSampleTest(qrels,data,0),modelname)

    np.savetxt(f"embeddings/demo-doc2vec-embedding-{CollectionName}", DocMatrix)

    with open(f"embeddings/demo-doc2vec-querymap-{CollectionName}", "w") as o:
        o.write(json.dumps(DocMap))
    # print(DocMatrix.shape)
