import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .retriever import retrieve_top_k

def load_titles(path='D:\\smt7\\Skripsi\\TitleRecommender\\data\\Datasets.csv', encoding='latin1'):
    df = pd.read_csv(path, encoding='latin1')
    return df['Judul'].tolist()

def get_recommendations(input_text, top_k=5, model='sbert'):
    titles = load_titles()
    
    if model == 'sbert':
        with open('./data/embeddings_sbert.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        sbert = SentenceTransformer('paraphrase-mpnet-base-v2')
        query_embedding = sbert.encode(input_text)

    elif model == 'tfidf':
        with open('./models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('./data/embeddings_tfidf.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        query_embedding = vectorizer.transform([input_text])

    else:
        raise ValueError("Model harus 'sbert' atau 'tfidf'")

    top_indices, scores = retrieve_top_k(query_embedding, embeddings, top_k)
    return [titles[i] for i in top_indices]
