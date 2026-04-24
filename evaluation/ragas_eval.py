from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

from sentence_transformers import SentenceTransformer
from ragas.embeddings import LangchainEmbeddingsWrapper

import os


# ---------- LLM (Groq) ----------
def get_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_llm.n = 1  # critical fix

    return wrapped_llm


# ---------- EMBEDDINGS (LOCAL, NO OPENAI) ----------
def get_embeddings():
    model = SentenceTransformer("BAAI/bge-small-en")

    class CustomEmbedding:
        def embed_documents(self, texts):
            return model.encode(texts, normalize_embeddings=True)

        def embed_query(self, text):
            return model.encode([text], normalize_embeddings=True)[0]

    return LangchainEmbeddingsWrapper(CustomEmbedding())


# ---------- DATA ----------
def prepare_ragas_data(query, answer, contexts, ground_truth):
    return Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })


# ---------- MAIN ----------
def run_ragas_evaluation(query, answer, retrieved_docs, ground_truth):
    contexts = [doc["text"] for doc in retrieved_docs]

    dataset = prepare_ragas_data(
        query=query,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    llm = get_llm()
    embeddings = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,   
        raise_exceptions=False
    )

    return result