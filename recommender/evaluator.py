import numpy as np
from sklearn.metrics import ndcg_score

def precision_at_k(y_true, y_pred, k):
    relevant = set(y_true)
    top_k = y_pred[:k]
    return len(set(top_k) & relevant) / k

def recall_at_k(y_true, y_pred, k):
    relevant = set(y_true)
    top_k = y_pred[:k]
    return len(set(top_k) & relevant) / len(relevant) if relevant else 0

def ndcg_at_k(y_true_binary, y_scores, k):
    """
    y_true_binary: list of 0/1 relevance for each candidate (same length as y_scores)
    y_scores: predicted similarity scores
    """
    return ndcg_score([y_true_binary], [y_scores], k=k)

def evaluate_recommender(test_queries, true_indices, embeddings, model, top_k=5):
    from recommender.retriever import retrieve_top_k
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(model)
    metrics = {'precision': [], 'recall': [], 'ndcg': []}

    for i, query in enumerate(test_queries):
        query_embedding = sbert.encode(query)
        top_indices, scores = retrieve_top_k(query_embedding, embeddings, k=top_k)
        
        # Prepare ground truth relevance
        relevant_index = true_indices[i]
        relevance = [1 if idx == relevant_index else 0 for idx in top_indices]
        
        metrics['precision'].append(precision_at_k([relevant_index], top_indices, k=top_k))
        metrics['recall'].append(recall_at_k([relevant_index], top_indices, k=top_k))
        metrics['ndcg'].append(ndcg_at_k(relevance, scores, k=top_k))

    return {k: np.mean(v) for k, v in metrics.items()}
