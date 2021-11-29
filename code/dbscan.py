from sklearn.cluster import DBSCAN
from dbscan_util import *
import numpy as np
import pandas as pd
import sys
import json

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


def runDBSCAN(
    epsilons,
    deltas,
    min_samples,
    embeddings,
    query_embeddings,
    dname,
    embedding,
    doc_map,
    query_map
):  

    MAPS = []
    RECALLS = []
    EPSILONS = []
    DELTAS = []
    TOTAL_CLUSTERS = []
    CLUSTERS_RETRIEVED = []
    NUM_DOCS_RETRIEVED = []

    logger.info(f"NUMBER OF QUERIES = {query_embeddings.shape[0]}")

    if len(epsilons) > 1:
        delta = deltas[0]
        for eps in epsilons:
            logger.info(f"COMPUTING CLUSTERING FOR EPS = {eps}, delta = {delta}")

            clustering = DBSCAN(
                eps=eps,
                metric='cosine',
                min_samples=2
            ).fit(embeddings)

            logger.info(f"NUMBER OF CLUSTERS FORMED = {len(set(clustering.labels_))}, EPS = {eps}, MIN_SAMPLES = {min_samples}")
                    
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
            EPSILONS.append(eps)
            DELTAS.append(delta)
            TOTAL_CLUSTERS.append(len(set(clustering.labels_)))
            CLUSTERS_RETRIEVED.append(clusters_retrieved)
            NUM_DOCS_RETRIEVED.append(num_docs_retrieved)
            logger.info(
                f"EPS = {eps}, DELTA = {delta}, MAP = {MAP}, RECALL = {recall}, TOTAL CLUSTERS = {len(set(clustering.labels_))}, CLUSTERS RETRIEVED = {clusters_retrieved}, NUM DOCS RETRIEVED = {num_docs_retrieved}")
        
        df = pd.DataFrame(
            {
                "EPSILONS": EPSILONS,
                "DELTAS": DELTAS,
                "MAPS": MAPS,
                "RECALLS": RECALLS,
                "TOTAL_CLUSTERS": TOTAL_CLUSTERS,
                "CLUSTERS_RETRIEVED": CLUSTERS_RETRIEVED,
                "NUM_DOCS_RETRIEVED": NUM_DOCS_RETRIEVED
            }
        )

        # df.to_csv(f"results/dbscan_{embedding}_{dname}_cosine_epsilon_extra", index=False)

    if len(deltas) > 1:
        eps = epsilons[0]
        for delta in deltas:
            logger.info(
                f"COMPUTING CLUSTERING FOR EPS = {eps}, delta = {delta}")

            clustering = DBSCAN(
                eps=eps,
                metric='cosine',
                min_samples=2
            ).fit(embeddings)

            logger.info(
                f"NUMBER OF CLUSTERS FORMED = {len(set(clustering.labels_))}, EPS = {eps}, MIN_SAMPLES = {min_samples}")

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
            EPSILONS.append(eps)
            DELTAS.append(delta)
            TOTAL_CLUSTERS.append(len(set(clustering.labels_)))
            CLUSTERS_RETRIEVED.append(clusters_retrieved)
            NUM_DOCS_RETRIEVED.append(num_docs_retrieved)
            logger.info(
                f"EPS = {eps}, DELTA = {delta}, MAP = {MAP}, RECALL = {recall}, TOTAL CLUSTERS = {len(set(clustering.labels_))}, CLUSTERS RETRIEVED = {clusters_retrieved}, NUM DOCS RETRIEVED = {num_docs_retrieved}")

        df = pd.DataFrame(
            {
                "EPSILONS": EPSILONS,
                "DELTAS": DELTAS,
                "MAPS": MAPS,
                "RECALLS": RECALLS,
                "TOTAL_CLUSTERS": TOTAL_CLUSTERS,
                "CLUSTERS_RETRIEVED": CLUSTERS_RETRIEVED,
                "NUM_DOCS_RETRIEVED": NUM_DOCS_RETRIEVED
            }
        )

        # df.to_csv(f"results/dbscan_{embedding}_{dname}_cosine_deltas_extra", index=False)
    

if __name__ == "__main__":
    embedding = sys.argv[1]
    dname = sys.argv[2]

    embeddings = np.loadtxt(f"embeddings/{embedding}-embedding-{dname}")
    query_embeddings = np.loadtxt(f"embeddings/{embedding}-query-embedding-{dname}")

    with open(f"embeddings/{embedding}-docmap-{dname}", "r") as f:
        doc_map = json.loads(f.read())
    with open(f"embeddings/{embedding}-querymap-{dname}", "r") as f:
        query_map = json.loads(f.read())

    logger.info(f"SHAPE OF DOCUMENT EMBEDDINGS = {embeddings.shape}")
    logger.info(f"SHAPE OF QUERY EMBEDDINGS = {query_embeddings.shape}")

    # min_samples = 2
    epsilons = [0.16]
    # epsilons = np.concatenate((np.linspace(0.0001, 0.2, num=6), np.linspace(0.3, 1, num=3)))
    deltas = [0, 0]
    # deltas = [0.07]
    # deltas = np.linspace(0, 0.2, num=10)

    runDBSCAN(
        epsilons=epsilons,
        deltas=deltas,
        min_samples=2,
        embeddings=embeddings,
        query_embeddings=query_embeddings,
        dname=dname,
        embedding=embedding,
        doc_map=doc_map,
        query_map=query_map
    )
