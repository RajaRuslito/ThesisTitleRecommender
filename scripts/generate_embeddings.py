import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender.embedder import generate_tfidf_embeddings, generate_sbert_embeddings
import pandas as pd

def main():
    df = pd.read_csv("D:\\smt7\\Skripsi\\TitleRecommender\\data\\Datasets.csv", encoding='latin1')
    titles = df['Judul'].tolist()

    print("Membuat TF-IDF embeddings...")
    generate_tfidf_embeddings(titles)

    print("Membuat SBERT embeddings...")
    generate_sbert_embeddings(titles)

    print("Semua embedding berhasil dibuat.")

if __name__ == "__main__":
    main()
