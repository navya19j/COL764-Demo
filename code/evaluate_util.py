import re
import numpy as np
from sklearn import cluster
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances

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

def getLabelVectors(
    doc_vectors, clustering
):  
    label_vectors = {}
    for label in set(clustering.labels_):
        label_vector = np.zeros(doc_vectors.shape[1])
        count = 0
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == label:
                label_vector += doc_vectors[i]
                count += 1
        label_vector /= count
        label_vectors[label] = label_vector
    return label_vectors

def findBestCluster(
    query_vector, label_vectors, metric
):
    best_cluster = -1
    smallest_distance = 1
    query_vector = np.array(query_vector)
    for cluster_id, label_vector in label_vectors.items():
        distance = pairwise_distances(
            query_vector.reshape(1, -1), label_vector.reshape(1, -1), metric=metric
            )[0][0]
        if distance < smallest_distance:
            smallest_distance = distance 
            best_cluster = cluster_id 
    
    return best_cluster


def loadScoresqrel(
    result_file
):  
    scores = {}
    with open(result_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            words = re.split("\t", line)
            # print(words)
            print(words)
            # print(line, result_file)
            query_id = words[0]
            doc_id = words[2]
            if not query_id in scores:
                scores[query_id] = []
            scores[query_id].append(doc_id)
    return scores

def loadScoresResults(
    result_file
):  
    scores = {}
    with open(result_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            words = re.split(" ", line)
            # print(words)
            print(words)
            # print(line, result_file)
            query_id = words[0]
            doc_id = words[1]
            if not query_id in scores:
                scores[query_id] = []
            scores[query_id].append(doc_id)
    return scores


def computeMAP(
    result_file, qrels_file
):
    qrels_file = "trec12-news.tsv"
    qrel_scores = loadScoresqrel(qrels_file)
    computed_scores = loadScoresResults(result_file)

    # print(qrel_scores)
    # print(computed_scores['51'])
    APs = []
    
    for query_id, doc_ids in computed_scores.items():
        # print(computed_scores[query_id])
        # print(qrel_scores[str(int(query_id))])

        a = computed_scores[query_id]
        b = qrel_scores[str(int(query_id))]
        print(len(set(a) & set(b)))

        relevant = 0
        total = 0
        precisions = []
        for doc_id in doc_ids:
            total += 1
            if doc_id in qrel_scores[str(int(query_id))]:
                relevant += 1
                precisions.append(relevant/total)
        if len(precisions) > 0:
            AP = np.mean(precisions)
        else:
            AP = 0
        APs.append(AP)
    
    if len(APs) == 0:
        return 0
    # APs = sorted(APs, reverse=True)[0:10]
    return np.mean(APs)
            


    # with open(result_file, "r") as f:
    #     lines = f.read().splitlines()
    #     for line in lines:
    #         words = re.split(" ", line)
    #         query_id = int(words[0])
    #         doc_id = int(words[2])
    #         if doc_id in scores[query_id]:
    #             good += 1
    #             print("SOMETHING IS GOOD!")

    #             APs.append(good / count)
    # if good == 0:
    #     print("NOTHING IS GOOD")
    # return np.mean(APs)


def computeMetrics(
    query_vectors, doc_vectors, clustering, doc_map, query_map, metric, dname, embedding
):  
    label_vectors = getLabelVectors(
        doc_vectors, clustering
    )

    logger.info(f"COMPUTING SIMILARITIES...")

    result_file = f"{embedding}-ranking-{dname}"
    open(f"rankings/{result_file}", "w").close()

    for i in range(len(query_vectors)):
        best_cluster = findBestCluster(
            query_vectors[i], label_vectors, metric=metric
        )
        scores = []
        for j in range(len(doc_vectors)):
            score = 0.0
            if clustering.labels_[j] != best_cluster:
                score = 0.0
            else:
                score = pairwise_distances(
                    query_vectors[i].reshape(1, -1), doc_vectors[j].reshape(1, -1), metric=metric
                )[0][0]
            scores.append((score, j))

        scores.sort(reverse=True)
        scores = scores[:min(len(scores), 100)]
        with open(f"rankings/{result_file}", "a") as f:
            for score, doc_id in scores:
                # print(i, doc_id)
                f.write(f"{query_map[str(i)]} 0 {doc_map[str(doc_id)]} {score}\n") 
    
    qrels_file = "trec12-news.tsv"
    MAP = computeMAP(
        qrels_file=qrels_file,
        result_file=result_file
    )

    return MAP, 0, 0, 0