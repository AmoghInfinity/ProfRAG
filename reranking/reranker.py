from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Cross-encoder for reranking retrieved documents
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank documents based on query relevance
        """
        if not documents:
            return []

        pairs = [(query, doc["text"]) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, _ in scored_docs[:top_k]]

        return reranked