import re
import numpy as np
from sklearn import cluster
from scipy.spatial.distance import cosine
import pytrec_eval
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("logs/agreement_all.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


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


def findGoodClusters(
    query_vector, label_vectors, delta
):
    query_vector = np.array(query_vector)
    similarities = []
    for cluster_id, label_vector in label_vectors.items():
        similarity = abs(1 - cosine(
            query_vector.reshape(1, -1), label_vector.reshape(1, -1)
        ))
        similarities.append((similarity, cluster_id))
    similarities.sort(reverse=True)
    # print(similarities[:10])
    good_clusters = [similarities[0][1]]
    for i in range(1, len(similarities)):
        if similarities[0][0] - similarities[i][0] > delta:
            break
        good_clusters.append(similarities[i][1])
    return good_clusters


def loadScores(
    result_file
):
    scores = {}
    with open(f"rankings/trec-format/{result_file}") as f:
        lines = f.read().splitlines()
        for line in lines:
            words = re.split(" |\t", line)
            query_id = str(int(words[0]))
            doc_id = words[2]
            if not query_id in scores:
                scores[query_id] = {}
            if len(words) > 4:
                scores[query_id][doc_id] = float(words[4])
            else:
                scores[query_id][doc_id] = 1
    return scores


def computeMAP(
    result_file, qrels_file
):
    qrel_scores = loadScores(qrels_file)
    computed_scores = loadScores(result_file)

    results = pytrec_eval.RelevanceEvaluator(
        qrel_scores, {'map'}).evaluate(computed_scores)

    MAPs = []
    for query_id, metrics in results.items():
        MAPs.append(metrics["map"])

    return np.mean(MAPs)


def computeRecall(
    result_file, qrels_file
):
    qrel_scores = loadScores(qrels_file)
    computed_scores = loadScores(result_file)

    results = pytrec_eval.RelevanceEvaluator(
        qrel_scores, {'recall.100'}).evaluate(computed_scores)

    recalls = []
    for query_id, metrics in results.items():
        recalls.append(metrics["recall_100"])

    return np.mean(recalls)


def computeMetrics(
    query_vectors, doc_vectors, clustering, doc_map, query_map, dname, embedding, delta
):
    label_vectors = getLabelVectors(
        doc_vectors, clustering
    )

    logger.info(f"COMPUTING SIMILARITIES...")

    result_file = f"kmeans-{embedding}-ranking-{dname}-extra"
    open(f"rankings/trec-format/{result_file}", "w").close()

    size = {}
    for label in clustering.labels_:
        if not label in size:
            size[label] = 0
        size[label] += 1

    total_sizes = []
    clusters_retrieveds = []

    for i in range(len(query_vectors)):
        good_clusters = findGoodClusters(
            query_vectors[i], label_vectors, delta
        )

        clusters_retrieveds.append(len(good_clusters))

        total_size = 0
        # tmp = []
        for cluster_id in good_clusters:
            total_size += size[cluster_id]
        #     tmp.append(size[cluster_id])
        # tmp.sort(reverse=True)
        # print(tmp[:10])
        total_sizes.append(total_size)

        logger.info(
            f"NUMBER OF CLUSTERS RETRIEVED = {len(good_clusters)}, TOTAL SIZE = {total_size}")
        scores = []
        for j in range(len(doc_vectors)):
            score = 0.0
            if clustering.labels_[j] not in good_clusters:
                score = 0.0
            else:
                score = abs(1 - cosine(
                    query_vectors[i].reshape(
                        1, -1), doc_vectors[j].reshape(1, -1)
                ))
            scores.append((score, j))

        scores.sort(reverse=True)
        scores = scores[:min(len(scores), 100)]
        with open(f"rankings/trec-format/{result_file}", "a") as f:
            for score, doc_id in scores:
                f.write(
                    f"{query_map[str(i)]} Q0 {doc_map[str(doc_id)]} 1 {score} runidl\n")
                pass

    qrels_file = "qrels-ap"

    MAP = computeMAP(
        qrels_file=qrels_file,
        result_file=result_file
    )

    recall = computeRecall(
        qrels_file=qrels_file,
        result_file=result_file
    )

    total_clusters = len(set(clustering.labels_))
    clusters_retrieved = np.mean(clusters_retrieveds)
    num_docs_retrieved = np.mean(total_sizes)

    return MAP, recall, total_clusters, clusters_retrieved, num_docs_retrieved
