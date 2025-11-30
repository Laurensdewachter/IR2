from storage import storage
import numpy as np
from bitarray import bitarray
from embedder import SentenceEmbedder
import pandas as pd
from pathlib import Path


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
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_plane(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        input_point = np.array(input_point)
        projections = np.dot(planes, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def index(self, input_point):
        # input_point = (doc_id, vector)
        doc_id, vector = input_point

        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Store full info
        value = (doc_id, tuple(vector))


        for i, table in enumerate(self.hash_tables):
            hash_value = self._hash(self.uniform_planes[i], vector)
            table.append_val(hash_value, value)

    def query(self, query_point, num_results=None, distance_func=None):
            """ Takes `query_point` which is either a tuple or a list of numbers,
            returns `num_results` of results as a list of tuples that are ranked
            based on the supplied metric function `distance_func`.

            :param query_point:
                A list, or tuple, or numpy ndarray that only contains numbers.
                The dimension needs to be 1 * `input_dim`.
                Used by :meth:`._hash`.
            :param num_results:
                (optional) Integer, specifies the max amount of results to be
                returned. If not specified all candidates will be returned as a
                list in ranked order.
            :param distance_func:
                (optional) The distance function to be used. Currently it needs to
                be one of ("hamming", "euclidean", "true_euclidean",
                "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
                will used.
            """

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
                    raise ValueError("The distance function name is invalid.")

                for i, table in enumerate(self.hash_tables):
                    binary_hash = self._hash(self.uniform_planes[i], query_point)
                    candidates.update(table.get_list(binary_hash))

            # rank candidates by distance function
            candidates = [((doc_id, vec), d_func(query_point, np.asarray(vec)))
              for (doc_id, vec) in candidates]

            candidates.sort(key=lambda x: x[1])

            return candidates[:num_results] if num_results else candidates

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)



if __name__ == "__main__":
    embedder = SentenceEmbedder()
    df_corpus, embeddings = embedder.encode_corpus()
    LSH = lsh(16, embeddings.shape[1], num_hashtables=10, storage_config={'dict': None})
    for idx, vec in enumerate(embeddings):
        LSH.index((idx, vec))
    
    query = embeddings[9645] # document 9645

    results = LSH.query(query, num_results=5, distance_func="l1norm")
    for (doc_id, _), dist in results:
        print(df_corpus.loc[doc_id, "plot"])