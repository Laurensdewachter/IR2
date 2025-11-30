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

from sklearn.cluster import KMeans

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

    # Perform KMeans clustering
    # TODO: Optimize parameters
    k = 1024
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(vectors)

    codebook = kmeans.cluster_centers_
    quantized_vectors = kmeans.predict(vectors)

    return quantized_vectors


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
        quantize_model(model)

    if args.lsh:
        if not model:
            raise ValueError("No model loaded for LSH indexing")
        lsh_index = lsh(hash_size=16, input_dim=model.wv.vectors.shape[1], num_hashtables=15, storage_config={'dict': None})
        
        start_time = time.time()
        for idx, vec in tqdm(enumerate(model.wv.vectors)):
            lsh_index.index((idx, vec))
        end_time = time.time()
        print(f"LSH indexing completed in {end_time - start_time:.2f} seconds.")

        print("Querying LSH...")
        query_vec = model.wv.vectors[0]

        start_time = time.time()
        results = lsh_index.query(query_vec, num_results=5, distance_func="euclidean")
        end_time = time.time()
        print(f"LSH query time: {end_time - start_time:.6f} seconds")

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

            