import os
import nltk
import gensim
import zipfile
import argparse

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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--training_data", "-t", type=str, help="Path to the data zip file to train on"
    )
    arg_parser.add_argument("--model", "-m", type=str, help="Path to a trained model")
    args = arg_parser.parse_args()

    if args.training_data:
        print(
            f"Training Word2Vec model on {os.path.splitext(args.training_data)[0]} zip-file..."
        )
        word2vec(args.training_data).save(
            f"{os.path.splitext(args.training_data)[0]}.model"
        )

    if args.model:
        print(f"Loading Word2Vec model from {args.model}...")
        model = gensim.models.Word2Vec.load(args.model)
