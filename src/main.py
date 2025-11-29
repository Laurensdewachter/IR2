import os
import nltk
import gensim
import zipfile
import argparse
import numpy as np

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
