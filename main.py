from ingestion.pdf_parser import extract_text_from_pdf
from chunking.context_chunker import chunk_documents
from embeddings.bge_embedder import BGEEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import CrossEncoderReranker
from generation.generator import Generator
from compression.context_compressor import ContextCompressor


def build_pipeline(file_path: str):
    """
    Build full pipeline once (extract → chunk → embed → index)
    """
    # Extract
    docs = extract_text_from_pdf(file_path)

    # Chunk
    chunks = chunk_documents(docs)

    # Embed
    embedder = BGEEmbedder()
    embedded_docs = embedder.embed_documents(chunks)

    # Retriever
    retriever = HybridRetriever()
    retriever.build_index(embedded_docs)

    # Reranker
    reranker = CrossEncoderReranker()

    # Generator
    generator = Generator()

    # Compressor (NEW)
    compressor = ContextCompressor()

    return retriever, embedder, reranker, generator, compressor


def query_pipeline(query: str, retriever, embedder, reranker, generator, compressor):
    """
    Full query pipeline (retrieve → rerank → compress → generate)
    """
    # Step 1: Retrieve more candidates
    results = retriever.retrieve(query, embedder, top_k=10)

    # Step 2: Rerank
    results = reranker.rerank(query, results)

    # Step 3: Compress context (NEW)
    results = compressor.compress(query, results)

    # Step 4: Generate answer
    answer = generator.generate_answer(query, results)

    return results, answer


if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"

    queries = [
        "What is SVM?",
        "advantages of svm",
        "types of svm",
        "what is classification algorithm"
    ]

    print("\n🔹 Building pipeline once...\n")

    retriever, embedder, reranker, generator, compressor = build_pipeline(file_path)

    for query in queries:
        print(f"\n\n🔍 Query: {query}\n")

        results, answer = query_pipeline(
            query,
            retriever,
            embedder,
            reranker,
            generator,
            compressor
        )

        print("\n--- Answer ---")
        print(answer)

        print("\n--- Top Sources ---")
        for r in results[:3]:
            print(r["metadata"])