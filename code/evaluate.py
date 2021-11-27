import re 
import numpy as np
from evaluate_util import *
import os

if __name__ == "__main__":

    dname = "trec12-news.tsv"
    rankings = os.listdir("Rankings")

    for rank_file in rankings:
        try:
            filepath = "Rankings/"+rank_file
            qrels_file = "trec12-news.tsv"
            print(rank_file)
            map_val = computeMAP(filepath, qrels_file)

            out_filename = filepath.split(".txt")[0]
            print(out_filename+"_MAP.txt")

            with open(out_filename,"w") as o:
                o.write(str(map_val))
        except:
            print("DS")
