from storage import storage
import numpy as np
from bitarray import bitarray
from embedder import SentenceEmbedder
import pandas as pd
from pathlib import Path


from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path.cwd().parent / "data"

# lshash: Locality Sensitive Hashing in Python
# disclaimer: This code has been modified from the original version
# 
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

class lsh(object):
    def __init__(self, hash_size, input_dim, num_hashtables=1, storage_config=None):
        
        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        self.embeddings = None
        
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):
        self.uniform_planes = [self._generate_uniform_plane() for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_plane(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        input_point = np.array(input_point)
        projections = np.dot(planes, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def index(self, input_point):
        doc_id, vector = input_point
        vector = vector.tolist() if isinstance(vector, np.ndarray) else vector
        value = (doc_id, tuple(vector))

        for i, table in enumerate(self.hash_tables):
            hash_value = self._hash(self.uniform_planes[i], vector)
            table.append_val(hash_value, value)

    def query(self, query_point, num_results=None, distance_func=None):
        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "hamming":
            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = lsh.hamming_dist(key, binary_hash)
                    if distance < 2:
                        candidates.update(table.get_list(key))
            d_func = lsh.euclidean_dist_square

        else:
            if distance_func == "euclidean":
                d_func = lsh.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = lsh.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = lsh.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = lsh.cosine_dist
            elif distance_func == "l1norm":
                d_func = lsh.l1norm_dist
            else:
                raise ValueError("Invalid distance name.")

            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                candidates.update(table.get_list(binary_hash))

        candidates = [((doc_id, vec), d_func(query_point, np.asarray(vec)))
                      for (doc_id, vec) in candidates]

        return sorted(candidates, key=lambda x: x[1])[:num_results]

    # ==============================
    #   Distance functions
    # ==============================

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

    # ==============================
    #   Automatic (k, L) selection
    # ==============================

    @staticmethod
    def _single_bit_collision_prob(s):
        return 1 - (np.arccos(s) / np.pi)

    @staticmethod
    def _solve_L(a2, target_p2):
        return int(np.ceil(np.log(1 - target_p2) / np.log(1 - a2)))

    @classmethod
    def auto(cls, input_dim, s1=0.3, s2=0.8, p1=0.1, p2=0.95, k_range=range(4, 25)):
        p_s1 = cls._single_bit_collision_prob(s1)
        p_s2 = cls._single_bit_collision_prob(s2)

        best = None

        for k in k_range:
            a1 = p_s1 ** k
            a2 = p_s2 ** k
            try:
                L = cls._solve_L(a2, p2)
            except:
                continue

            fp = 1 - (1 - a1)**L

            if fp <= p1:
                cost = L * k
                if best is None or cost < best[3]:
                    best = (k, L, fp, cost)

        if best is None:
            raise ValueError("No valid (k,L) satisfying constraints.")

        k, L, fp, _ = best

        print("=== Auto-selected LSH parameters ===")
        print(f"k (bits per table): {k}")
        print(f"L (number of tables): {L}")
        print(f"False positive rate: {fp:.4f}")
        print("====================================")

        return cls(hash_size=k, input_dim=input_dim, num_hashtables=L)





def ground_truth_top_k(query_vecs, embeddings, k=10):
    sims = cosine_similarity(query_vecs, embeddings)
    return np.argsort(-sims, axis=1)[:, :k]

def lsh_top_k(query_vecs, LSH, k=10, distance_func="euclidean"):
    lsh_results = []
    for q in query_vecs:
        res = LSH.query(q, num_results=k, distance_func=distance_func)
        retrieved_ids = [doc_id for (doc_id, _), dist in res]
        retrieved_ids = (retrieved_ids + [-1]*k)[:k]
        lsh_results.append(retrieved_ids)
    return np.array(lsh_results)

def precision_recall_at_k(gt, pred, k=10):
    precisions = []
    recalls = []
    for g, p in zip(gt, pred):
        g_set = set(g[:k])
        p_set = set(p[:k])
        intersection = len(g_set & p_set)
        precisions.append(intersection / k)
        recalls.append(intersection / len(g_set))
    return np.mean(precisions), np.mean(recalls)

def evaluate_lsh_grid(query_vecs, embeddings, hash_sizes, num_tables_list, distance_funcs, k=10):
    results = []
    for hs in hash_sizes:
        for nt in num_tables_list:
            for dist in distance_funcs:
                print(f"Evaluating hash_size={hs}, num_tables={nt}, distance_func={dist}")

                LSH = lsh(hash_size=hs,
                          input_dim=embeddings.shape[1],
                          num_hashtables=nt)

                for idx, vec in enumerate(embeddings):
                    LSH.index((idx, vec))

                gt = ground_truth_top_k(query_vecs, embeddings, k=k)
                pred = lsh_top_k(query_vecs, LSH, k=k, distance_func=dist)

                p, r = precision_recall_at_k(gt, pred, k=k)
                
                results.append({
                    "hash_size": hs,
                    "num_tables": nt,
                    "distance_func": dist,
                    "precision@10": p,
                    "recall@10": r
                })

                # print(f" --> Precision@{k}: {p:.4f}, Recall@{k}: {r:.4f}\n")
    return results



if __name__ == "__main__":
    embedder = SentenceEmbedder()
    df_corpus, embeddings = embedder.encode_corpus()
    
    print(embeddings.shape) 
    
    query_vecs = embeddings[10000:10100]  # 100 queries

    # --- Grid search hyperparameters ---
    hash_sizes = [16, 20, 22, 24]
    num_tables_list = [5, 10, 15]
    distance_funcs = ["cosine", "euclidean", "hamming"]

    grid_results = evaluate_lsh_grid(query_vecs, embeddings, hash_sizes, num_tables_list, distance_funcs, k=10)
    
    # Convert results to DataFrame for easy analysis
    df_results = pd.DataFrame(grid_results)
    print(df_results.sort_values(by="precision@10", ascending=False))

    