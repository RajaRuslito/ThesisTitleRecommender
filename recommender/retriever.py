import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq

def retrieve_top_k(embedding_query, embedding_corpus, k=5):
    sim_scores = cosine_similarity(embedding_query, embedding_corpus)[0]
    top_indices = np.argsort(sim_scores)[::-1][:k]
    return top_indices, sim_scores[top_indices]
