import os
import nltk
import gensim
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex
import time
from lsh import lsh
from lsh import evaluate_lsh_grid
from sklearn.cluster import KMeans
import pandas as pd
import math

nltk.download("punkt_tab", quiet=True)


def word2vec(data_path: str) -> gensim.models.Word2Vec:
    """
    Based on the tutorial from GeeksforGeeks (https://www.geeksforgeeks.org/python/python-word-embedding-using-word2vec/)

    :param data_path: Zip file containing text data
    :return: Trained Word2Vec model
    """
    model = None
    with zipfile.ZipFile(data_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            with zip_ref.open(file) as f:
                # Reading and cleaning the text
                content = f.read().decode("utf-8", errors="ignore")
                cleaned_text = content.replace("\n", " ")

                # Tokenization
                data = []
                for sentence in nltk.sent_tokenize(cleaned_text):
                    data.append(nltk.word_tokenize(sentence))

                # Training the Word2Vec model
                if not model:
                    model = gensim.models.Word2Vec(
                        data,
                        min_count=1,
                        vector_size=100,
                        window=5,
                        workers=4,
                        sg=1,
                    )
                else:
                    model.train(data, total_examples=1, epochs=1)
    return model


def quantize_model(model: gensim.models.Word2Vec):
    """
    Quantizes the Word2Vec model's word vectors

    :param model: Quantized Word2Vec model
    """
    # Load vectors and vocabulary
    vectors = model.wv.vectors
    vocabulary = model.wv.index_to_key
    K = int(vectors.shape[0] / 512)


    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)
    centroids = kmeans.cluster_centers_
    posting_lists = {i: [] for i in range(K)}

    for idx, label in enumerate(labels):
        posting_lists[label].append(idx)
    return posting_lists, centroids


def search(centroids, posting_lists, embeddings, query_vec, top_m=5):
    # 1. Find nearest centroid (coarse search)
    distances = np.linalg.norm(centroids - query_vec, axis=1)
    nearest_cluster = np.argmin(distances)

    # 2. Fine search in posting list
    candidate_ids = posting_lists[nearest_cluster]
    candidate_vecs = embeddings[candidate_ids]

    fine_dists = np.linalg.norm(candidate_vecs - query_vec, axis=1)
    top_idx = np.argsort(fine_dists)[:top_m]

    return [(candidate_ids[i], fine_dists[i]) for i in top_idx]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--training_data", "-t", type=str, help="Path to the data zip file to train on"
    )
    arg_parser.add_argument("--model", "-m", type=str, help="Path to a trained model")
    arg_parser.add_argument(
        "--quantize", "-q", action="store_true", help="Enable model quantization"
    )

    arg_parser.add_argument("--annoy", action="store_true", help="Enable Annoy indexing")

    arg_parser.add_argument("--lsh", action="store_true", help="Enable LSH indexing",)

    args = arg_parser.parse_args()

    model = None
    if args.training_data:
        print(
            f"Training Word2Vec model on {os.path.splitext(args.training_data)[0]} zip-file..."
        )
        model = word2vec(args.training_data)
        model.save(f"{os.path.splitext(args.training_data)[0]}.model")

    if args.model:
        print(f"Loading Word2Vec model from {args.model}...")
        model = gensim.models.Word2Vec.load(args.model)

    if args.quantize:
        if not model:
            raise ValueError("No model loaded to quantize.")
        if not args.model:
            raise ValueError("Model path must be provided for quantization.")

        print("Quantizing the model...")
        start_time = time.time()
        posting_lists, centroids = quantize_model(model)
        end_time = time.time()
        print(f"Quantization completed in {end_time - start_time:.6f} seconds")

        # Run a sample search
        search_vec = model.wv.vectors[0]
        start_time = time.time()
        results = search(centroids, posting_lists, model.wv.vectors, search_vec, top_m=5)
        end_time = time.time()
        print(f"Quantized search completed in {end_time - start_time:.6f} seconds\n")

        print("Top 5 search results (doc_id, distance):")
        for doc_id, dist in results:
            print(f"Doc ID: {doc_id}, Distance: {dist}")


    if args.lsh:
        if not model:
            raise ValueError("No model loaded for LSH indexing")

        print("Preparing embeddings for LSH...")
        embeddings = model.wv.vectors
        input_dim = embeddings.shape[1]

        # Create 100 query vectors
        query_vecs = embeddings[:100]

        # ---------------------------
        # 1. Grid Search LSH to find best k
        # ---------------------------

        grid_results = evaluate_lsh_grid(
            query_vecs,
            embeddings,
            hash_sizes=[16, 20, 22, 24],
            num_tables_list=[5, 10, 15],
            distance_funcs=["cosine", "euclidean", "hamming"],
            k=10
        )
        df_results = pd.DataFrame(grid_results)
        print("\nGrid search results:")
        print(df_results)

        # Select best k based on precision@10
        best_row = df_results.loc[df_results["precision@10"].idxmax()]
        best_k = best_row["hash_size"]
        distance_func = best_row["distance_func"]
        print(f"\nBest k from grid search: {best_k}, using distance_func: {distance_func}")

        # ---------------------------
        # 2. Compute theoretical L using auto formula
        # ---------------------------
        s2 = 0.8   # similarity threshold for true neighbors
        p2 = 0.95  # probability to capture true neighbors

        p_s2 = lsh._single_bit_collision_prob(s2)
        a2 = p_s2 ** best_k
        L = lsh._solve_L(a2, p2)
        print(f"Theoretical number of tables L: {L}")

        # ---------------------------
        # 3. Build hybrid LSH index
        # ---------------------------
        print("\n=== Building Hybrid LSH Index ===")
        LSH = lsh(hash_size=best_k, input_dim=input_dim, num_hashtables=L)

        print("Indexing embeddings into LSH...")
        start_time = time.time()
        for idx, vec in enumerate(embeddings):
            LSH.index((idx, vec))
        end_time = time.time()
        print(f"Indexing completed in {end_time - start_time:.6f} seconds")

        # ---------------------------
        # 4. Query the hybrid LSH
        # ---------------------------
        print("\n=== Querying Hybrid LSH ===")
        query_vec = embeddings[0]

        start_time = time.time()
        results = LSH.query(query_vec, num_results=5, distance_func="hamming")
        end_time = time.time()

        print(f"Query time: {end_time - start_time:.6f} seconds\n")
        print("Top 5 results (doc_id, distance):")
        for (doc_id, _), dist in results:
            print(f"Doc ID: {doc_id}, Distance: {dist}")




    if args.annoy:
        if not model:
            raise ValueError("No model loaded for Annoy indexing.")
        length = model.wv.vectors.shape[1]
        t = AnnoyIndex(length, 'angular')
        
        start_time = time.time()
        for i in tqdm(range(len(model.wv.vectors))):
            t.add_item(i, model.wv.vectors[i])
        t.build(10)
        t.save('annoy.ann')
        end_time = time.time()
        print(f"Annoy indexing completed in {end_time - start_time:.2f} seconds.")

        print("Querying Annoy...")
        u = AnnoyIndex(length, 'angular')
        u.load('annoy.ann')
        query_vec = model.wv.vectors[0]

        start_time = time.time()
        indices = u.get_nns_by_vector(query_vec, 5, include_distances=True)
        end_time = time.time()
        print(f"Annoy query time: {end_time - start_time:.6f} seconds")

        print("Top 5 results (doc_id, distance):")
        for doc_id, dist in zip(*indices):
            print(f"Doc ID: {doc_id}, Distance: {dist}")

    