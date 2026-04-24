from ingestion.pdf_parser import extract_text_from_pdf
from chunking.context_chunker import chunk_documents
from embeddings.bge_embedder import BGEEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import CrossEncoderReranker
from generation.generator import Generator
from compression.context_compressor import ContextCompressor
from evaluation.ragas_eval import run_ragas_evaluation


def build_pipeline(file_path: str):
    docs = extract_text_from_pdf(file_path)
    chunks = chunk_documents(docs)

    embedder = BGEEmbedder()
    embedded_docs = embedder.embed_documents(chunks)

    retriever = HybridRetriever()
    retriever.build_index(embedded_docs)

    reranker = CrossEncoderReranker()
    generator = Generator()
    compressor = ContextCompressor()

    return retriever, embedder, reranker, generator, compressor


def query_pipeline(query: str, retriever, embedder, reranker, generator, compressor):
    results = retriever.retrieve(query, embedder, top_k=10)
    results = reranker.rerank(query, results)
    results = compressor.compress(query, results)

    answer = generator.generate_answer(query, results)

    return results, answer


def print_scores(scores):
    """
    Safe printing for RAGAS EvaluationResult
    Handles float, None, and string values
    """

    try:
        scores_dict = scores.to_pandas().iloc[0].to_dict()
    except Exception:
        print("Failed to parse evaluation result")
        return

    print("Evaluation Scores:")

    for key, value in scores_dict.items():

        if value is None:
            print(f"{key}: None")
            continue

        # Try to convert to float
        try:
            numeric_value = float(value)
            print(f"{key}: {round(numeric_value, 4)}")
        except (ValueError, TypeError):
            # fallback for strings or unexpected values
            print(f"{key}: {value}")

def compute_average_scores(all_scores):
    """
    Compute average only for numeric RAGAS metrics
    """

    metric_keys = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    avg_scores = {}

    for key in metric_keys:
        values = []

        for score in all_scores:
            try:
                score_dict = score.to_pandas().iloc[0].to_dict()
                value = score_dict.get(key)

                if value is None:
                    continue

                # Ensure numeric
                value = float(value)
                values.append(value)

            except Exception:
                continue

        if values:
            avg_scores[key] = sum(values) / len(values)
        else:
            avg_scores[key] = None

    return avg_scores

if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"

    queries = [
        "What is SVM?",
        "advantages of svm",
        "types of svm",
        "what is classification algorithm"
    ]

    ground_truths = {
        "What is SVM?": "SVM is a supervised learning algorithm used for classification and regression.",
        "advantages of svm": "SVM works well in high dimensional spaces, avoids overfitting, and provides accurate classification.",
        "types of svm": "Types of SVM include linear SVM and non-linear SVM.",
        "what is classification algorithm": "A classification algorithm is a supervised learning method used to categorize data into classes."
    }

    print("\nBuilding pipeline...\n")

    retriever, embedder, reranker, generator, compressor = build_pipeline(file_path)

    all_scores = []

    for query in queries:
        print(f"\nQuery: {query}\n")

        results, answer = query_pipeline(
            query,
            retriever,
            embedder,
            reranker,
            generator,
            compressor
        )

        print("Answer:")
        print(answer)

        print("\nTop Sources:")
        for r in results[:3]:
            print(r["metadata"])

        print("\nRunning RAGAS Evaluation...\n")

        eval_result = run_ragas_evaluation(
            query=query,
            answer=answer,
            retrieved_docs=results,
            ground_truth=ground_truths.get(query, "")
        )

        print_scores(eval_result)

        all_scores.append(eval_result)

    print("\nFinal Average Scores:\n")

    avg_scores = compute_average_scores(all_scores)

    print_scores(avg_scores)