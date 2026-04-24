from typing import List, Dict
import numpy as np
import faiss
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self):
        self.documents = []
        self.texts = []
        self.embeddings = None
        self.index = None
        self.bm25 = None

    def build_index(self, embedded_docs: List[Dict]):
        """
        Build FAISS + BM25 index
        """
        self.documents = embedded_docs
        self.texts = [doc["text"] for doc in embedded_docs]

        embeddings = np.array(
            [doc["embedding"] for doc in embedded_docs]
        ).astype("float32")

        self.embeddings = embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, embedder, top_k: int = 5, alpha: float = 0.7):
        """
        Hybrid retrieval
        """
        query_emb = embedder.embed_texts([query])[0]
        query_emb = np.array([query_emb]).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)

        vector_scores = {idx: score for idx, score in zip(indices[0], scores[0])}

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-6)

        combined_scores = {}

        for i in range(len(self.documents)):
            v_score = vector_scores.get(i, 0)
            b_score = bm25_scores[i]

            combined_scores[i] = alpha * v_score + (1 - alpha) * b_score

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [self.documents[i] for i, _ in ranked[:top_k]]