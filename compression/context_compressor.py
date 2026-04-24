from typing import List, Dict
import re


class ContextCompressor:
    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def score_sentence(self, sentence: str, query: str) -> int:
        """
        Simple keyword overlap scoring
        """
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())

        return len(query_words.intersection(sentence_words))

    def compress(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Compress documents by selecting most relevant sentences
        """
        compressed_docs = []

        for doc in documents:
            sentences = self.split_sentences(doc["text"])

            # Score sentences
            scored = [
                (sentence, self.score_sentence(sentence, query))
                for sentence in sentences
            ]

            # Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take top sentences
            top_sentences = [s for s, _ in scored[:self.max_sentences]]

            compressed_text = " ".join(top_sentences)

            if compressed_text:
                compressed_docs.append({
                    "text": compressed_text,
                    "metadata": doc["metadata"]
                })

        return compressed_docs