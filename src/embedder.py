from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd

DATA_DIR = Path.cwd().parent / "data"

class SentenceEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_corpus(self, data_dir=DATA_DIR):
        def csv_to_df():
            corpus = []
            for file_path in data_dir.iterdir():
                df = pd.read_csv(file_path)
                corpus.append(df)
            return pd.concat(corpus).sort_values(by="title").reset_index(drop=True)
        
        df_corpus = csv_to_df()
        embeddings = self.model.encode(df_corpus["plot"].to_list())
        return df_corpus, embeddings
