from typing import List, Dict
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    _model = None  # shared model across instances

    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        """
        Initialize BGE embedding model (singleton-style).
        """
        if BGEEmbedder._model is None:
            BGEEmbedder._model = SentenceTransformer(model_name)

        self.model = BGEEmbedder._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate normalized embeddings for a list of texts.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Attach embeddings to documents.
        """
        if not documents:
            return []

        texts = [doc.get("text", "") for doc in documents]
        embeddings = self.embed_texts(texts)

        embedded_docs = []

        for doc, emb in zip(documents, embeddings):
            embedded_docs.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "embedding": emb.tolist()
            })

        return embedded_docs


if __name__ == "__main__":
    from ingestion.pdf_parser import extract_text_from_pdf
    from ingestion.ppt_parser import extract_text_from_ppt
    from chunking.context_chunker import chunk_documents

    pdf_path = "data/raw/sample.pdf"
    ppt_path = "data/raw/sample.pptx"

    print("\n🔹 Running Full Pipeline (PDF → Chunk → Embed)\n")

    embedder = BGEEmbedder()  # load once

    # --- PDF ---
    pdf_docs = extract_text_from_pdf(pdf_path)
    pdf_chunks = chunk_documents(pdf_docs)
    embedded_pdf = embedder.embed_documents(pdf_chunks)

    for doc in embedded_pdf[:2]:
        print("\n--- PDF Embedded ---")
        print(doc["text"][:150])
        print(doc["metadata"])
        print(f"Embedding dim: {len(doc['embedding'])}")

    # --- PPT ---
    ppt_docs = extract_text_from_ppt(ppt_path)
    ppt_chunks = chunk_documents(ppt_docs)
    embedded_ppt = embedder.embed_documents(ppt_chunks)

    for doc in embedded_ppt[:2]:
        print("\n--- PPT Embedded ---")
        print(doc["text"][:150])
        print(doc["metadata"])
        print(f"Embedding dim: {len(doc['embedding'])}")