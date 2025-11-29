import argparse
import gensim
import nltk
import zipfile

nltk.download("punkt_tab")


def word2vec(data_path: str) -> gensim.models.Word2Vec:
    """
    Based on the tutorial from GeeksforGeeks (https://www.geeksforgeeks.org/python/python-word-embedding-using-word2vec/)

    :param data_path: ZIP file containing text data
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
                        data, min_count=1, vector_size=100, window=5, workers=4, sg=1
                    )
                else:
                    model.train(data, total_examples=1, epochs=1)
    return model


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--train", "-t", type=str, help="Path to the data zip file to train on"
    )
    args = arg_parser.parse_args()

    word2vec(args.train).save("word2vec.model")
