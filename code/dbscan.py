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
    metric,
    min_samples,
    embeddings,
    query_embeddings,
    dname,
    embedding,
    doc_map,
    query_map
):  
    """ Run DBSCAN with given parameters
    Returns MAP
    """

    MAPs = []
    nDCG_5s = []
    nDCG_10s = []
    nDCG_50s = []
    EPSILONS = []
    CLUSTERS = []

    logger.info(f"NUMBER OF QUERIES = {query_embeddings.shape[0]}")

    for eps in epsilons:
        logger.info(f"COMPUTING CLUSTERING FOR EPS = {eps}")

        clustering = DBSCAN(
            eps=eps,
            metric='cosine',
            min_samples=2
        ).fit(embeddings)

        logger.info(f"NUMBER OF CLUSTERS FORMED = {len(set(clustering.labels_))}, EPS = {eps}, MIN_SAMPLES = {min_samples}")
        
        CLUSTERS.append(len(set(clustering.labels_)))
        
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

        MAP, nDCG_5, nDCG_10, nDCG_50 = computeMetrics(
            query_embeddings, embeddings, clustering, doc_map, query_map, metric, dname, embedding
        )

        MAPs.append(MAP)
        nDCG_5s.append(nDCG_5)
        nDCG_10s.append(nDCG_10)
        nDCG_50s.append(nDCG_50)
        EPSILONS.append(eps)

        logger.info(f"EPS = {eps}, MAP = {MAP}")

    df = pd.DataFrame(
        {
            "EPSILONS": EPSILONS,
            "MAPs": MAPs,
            "nDCG_5s": nDCG_5s,
            "nDCG_10s": nDCG_10s,
            "nDCG_50s": nDCG_50s,
            "CLUSTERS": CLUSTERS
        }
    )

    df.to_csv(f"results/dbscan_{embedding}_{dname}_{metric})_double_check", index=False)
    

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

    metric = "cosine"
    min_samples = 2
    epsilons = np.concatenate((np.linspace(0.05, 0.2, num=6), np.linspace(0.3, 1, num=3)))

    runDBSCAN(
        epsilons=epsilons,
        metric=metric,
        min_samples=2,
        embeddings=embeddings,
        query_embeddings=query_embeddings,
        dname=dname,
        embedding=embedding,
        doc_map=doc_map,
        query_map=query_map
    )
