import ir_datasets
import re 
import numpy as np
import os
from sklearn.metrics import average_precision_score

# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# ch.setFormatter(formatter)
# logger.addHandler(ch)
# fh = logging.FileHandler("logs/agreement_all.log")
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

def load_scores(dname):
    scores = {}
    file = open(dname)
    lines = file.readlines()
    rels = " ".join(lines)
    qrels = re.split('''[\n]''',rels)

    for qrel in qrels:
        qrel_desc = re.split('''[\t]''',qrel)
        if (len(qrel_desc)>3):
            query_id = int(qrel_desc[0])
            doc_id = qrel_desc[2]
            if not query_id in scores:
                scores[query_id] = {}
            scores[query_id][doc_id] = int(qrel_desc[-1])

    return scores 

def computeMAP(
    dname,
    result_file
):  
    map_dict = {}
    APs = []
    scores = load_scores(dname)
    good = 0
    count = 0
    y_true = []
    y_pred = []
    # threshold = 100
    with open(result_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            words = re.split(" ", line)
            query_id = int(words[0])
            doc_id = words[1]
            score = float(words[2])

            if query_id in map_dict:
                y_pred.append(score)
                if doc_id in scores[query_id] and scores[query_id][doc_id] > 0:
                    y_true.append(1)
                    count+=1
                else:
                    y_true.append(0)
            else:
                if (len(y_true)>0):
                    if (count>0):
                        print(query_id)
                        score_new = average_precision_score(y_true,y_pred)
                        APs.append(score_new)
                y_true = []
                y_pred = []
                y_pred.append(score)
                count = 0

                if doc_id in scores[query_id] and scores[query_id][doc_id] > 0:
                    y_true.append(1)
                    count+=1
                else:
                    y_true.append(0)
                map_dict[query_id] = True
    print(APs)
    return np.mean(APs)

def computeNDCG(
    dname,
    result_file
):     

    scores = load_scores(dname)
    gains = []
    DCGs = []
    iDCGs = []

    nDCG_5 = 0.0
    nDCG_10 = 0.0
    nDCG_50 = 0.0

    with open(result_file, "r") as f:
        lines = f.read().splitlines()
        prev_query = -1

        for line in lines:
            words = re.split(" ", line)
            query_id = int(words[0])
            doc_id = words[1]

            if query_id != prev_query and prev_query != -1:

                prev = 0
                for gain in gains:
                    prev += gain
                    DCGs.append(prev)
                
                prev = 0
                gains.sort(reverse=True)
                for gain in gains:
                    prev += gain 
                    iDCGs.append(prev)

                DCGs = np.array(DCGs)
                iDCGs = np.array(iDCGs)
                nDCGs = DCGs/iDCGs

                nDCG_5 += nDCGs[4]
                nDCG_10 += nDCGs[9]
                nDCG_50 += nDCGs[49]

                prev_query = query_id
                DCGs = []
                iDCGs = []
                gains = []

            if doc_id in scores[query_id]:
                gains.append(max(scores[query_id][doc_id], 0))
            else:
                gains.append(0)

        prev = 0
        for gain in gains:
            prev += gain
            DCGs.append(prev)
        
        prev = 0
        gains.sort(reverse=True)
        for gain in gains:
            prev += gain 
            iDCGs.append(prev)

        DCGs = np.array(DCGs)
        iDCGs = np.array(iDCGs)
        nDCGs = DCGs/iDCGs

        nDCG_5 += nDCGs[4]
        nDCG_10 += nDCGs[9]
        nDCG_50 += nDCGs[49]

        prev_query = query_id
        DCGs = []
        iDCGs = []
        gains = []

    nDCG_5 /= query_id
    nDCG_10 /= query_id
    nDCG_50 /= query_id     

    return nDCG_5, nDCG_10, nDCG_50

if __name__ == "__main__":

    dname = "trec12-news.tsv"
    result_file= os.path.join("Kmeans","K=50,10k,vec=5000","result.txt")
    dict = load_scores(dname)

    print(computeMAP(
        dname=dname,
        result_file=result_file
    ))

    # print(computeNDCG(
    #     dname=dname,
    #     result_file=result_file
    # ))
