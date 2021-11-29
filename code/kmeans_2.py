import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
import numpy as np

from kmeans_2_util import *
import numpy as np
import pandas as pd
import sys
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


def runKMEANS(
    cluster_counts,
    deltas,
    embeddings,
    query_embeddings,
    dname,
    embedding,
    doc_map,
    query_map
):

    MAPS = []
    RECALLS = []
    DELTAS = []
    CLUSTER_COUNTS = []
    CLUSTERS_RETRIEVED = []
    NUM_DOCS_RETRIEVED = []

    logger.info(f"NUMBER OF QUERIES = {query_embeddings.shape[0]}")

    if len(cluster_counts) > 1:
        delta = deltas[0]
        for cluster_count in cluster_counts:
            logger.info(
                f"COMPUTING CLUSTERING FOR CLUSTER COUNT = {cluster_count}, delta = {delta}")

            clustering = KMeans(n_clusters=int(cluster_count)).fit(embeddings)

            logger.info(
                f"NUMBER OF CLUSTERS FORMED = {len(set(clustering.labels_))}, CLUSTER COUNT = {cluster_count}")

            size = {}
            for label in clustering.labels_:
                if not label in size:
                    size[label] = 0
                size[label] += 1

            sizes = []
            for k, v in size.items():
                sizes.append((v, k))
            sizes.sort(reverse=True)

            for i in range(min(10, len(sizes))):
                logger.info(f"LABEL = {sizes[i][1]}, SIZE = {sizes[i][0]}")

            MAP, recall, total_clusters, clusters_retrieved, num_docs_retrieved = computeMetrics(
                query_embeddings, embeddings, clustering, doc_map, query_map, dname, embedding, delta
            )

            MAPS.append(MAP)
            RECALLS.append(recall)
            DELTAS.append(delta)
            CLUSTER_COUNTS.append(len(set(clustering.labels_)))
            CLUSTERS_RETRIEVED.append(clusters_retrieved)
            NUM_DOCS_RETRIEVED.append(num_docs_retrieved)
            logger.info(
                f"CLUSTER COUNT = {cluster_count}, DELTA = {delta}, MAP = {MAP}, RECALL = {recall}, CLUSTERS RETRIEVED = {clusters_retrieved}, NUM DOCS RETRIEVED = {num_docs_retrieved}")

        df = pd.DataFrame(
            {
                "CLUSTER_COUNTS": CLUSTER_COUNTS,
                "DELTAS": DELTAS,
                "MAPS": MAPS,
                "RECALLS": RECALLS,
                "CLUSTERS_RETRIEVED": CLUSTERS_RETRIEVED,
                "NUM_DOCS_RETRIEVED": NUM_DOCS_RETRIEVED
            }
        )

        # df.to_csv(f"results/kmeans_{embedding}_{dname}_cosine_cluster_counts_extra", index=False)

    if len(deltas) > 1:
        cluster_count = cluster_counts[0]
        for delta in deltas:
            logger.info(f"DELTA = {delta}")
            logger.info(
                f"COMPUTING CLUSTERING FOR CLUSTER COUNT = {cluster_count}, delta = {delta}")

            model = KMeans(n_clusters=int(cluster_count), init='k-means++')
            clustering = model.fit(embeddings)

            logger.info(
                f"NUMBER OF CLUSTERS FORMED = {len(set(clustering.labels_))}, CLUSTER COUNT = {cluster_count}")

            size = {}
            for label in clustering.labels_:
                if not label in size:
                    size[label] = 0
                size[label] += 1

            sizes = []
            for k, v in size.items():
                sizes.append((v, k))
            sizes.sort(reverse=True)

            for i in range(min(10, len(sizes))):
                logger.info(f"LABEL = {sizes[i][1]}, SIZE = {sizes[i][0]}")

            MAP, recall, total_clusters, clusters_retrieved, num_docs_retrieved = computeMetrics(
                query_embeddings, embeddings, clustering, doc_map, query_map, dname, embedding, delta
            )

            MAPS.append(MAP)
            RECALLS.append(recall)
            DELTAS.append(delta)
            CLUSTER_COUNTS.append(len(set(clustering.labels_)))
            CLUSTERS_RETRIEVED.append(clusters_retrieved)
            NUM_DOCS_RETRIEVED.append(num_docs_retrieved)
            logger.info(
                f"CLUSTER COUNT = {cluster_count}, DELTA = {delta}, MAP = {MAP}, RECALL = {recall}, CLUSTERS RETRIEVED = {clusters_retrieved}, NUM DOCS RETRIEVED = {num_docs_retrieved}")

        df = pd.DataFrame(
            {
                "CLUSTER_COUNTS": CLUSTER_COUNTS,
                "DELTAS": DELTAS,
                "MAPS": MAPS,
                "RECALLS": RECALLS,
                "CLUSTERS_RETRIEVED": CLUSTERS_RETRIEVED,
                "NUM_DOCS_RETRIEVED": NUM_DOCS_RETRIEVED
            }
        )

        df.to_csv(
            f"results/kmeans_{embedding}_{dname}_cosine_deltas_extra", index=False)

if __name__ == "__main__":
    embedding = sys.argv[1]
    dname = sys.argv[2]

    embeddings = np.loadtxt(f"embeddings/{embedding}-embedding-{dname}")
    query_embeddings = np.loadtxt(
        f"embeddings/{embedding}-query-embedding-{dname}")

    with open(f"embeddings/{embedding}-docmap-{dname}", "r") as f:
        doc_map = json.loads(f.read())
    with open(f"embeddings/{embedding}-querymap-{dname}", "r") as f:
        query_map = json.loads(f.read())

    logger.info(f"SHAPE OF DOCUMENT EMBEDDINGS = {embeddings.shape}")
    logger.info(f"SHAPE OF QUERY EMBEDDINGS = {query_embeddings.shape}")

    # cluster_counts = np.linspace(50, 250, num=5)
    cluster_counts = [50]
    deltas = [0.07, 0.07]
    # deltas = [0.07]
    # deltas = np.linspace(0, 0.2, num=10)

    runKMEANS(
        cluster_counts=cluster_counts,
        deltas=deltas,
        embeddings=embeddings,
        query_embeddings=query_embeddings,
        dname=dname,
        embedding=embedding,
        doc_map=doc_map,
        query_map=query_map
    )
