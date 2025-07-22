import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def generate_tfidf_embeddings(titles, save_vectorizer='./models/tfidf_vectorizer.pkl', save_embeddings='./data/embeddings_tfidf.pkl'):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(titles)

    os.makedirs(os.path.dirname(save_vectorizer), exist_ok=True)
    os.makedirs(os.path.dirname(save_embeddings), exist_ok=True)

    with open(save_vectorizer, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(save_embeddings, 'wb') as f:
        pickle.dump(X, f)

    print("[INFO] TF-IDF embeddings saved")

def generate_sbert_embeddings(titles, model_name='paraphrase-mpnet-base-v2', save_path='./data/embeddings_sbert.pkl'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles, show_progress_bar=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print("[INFO] SBERT embeddings saved")

