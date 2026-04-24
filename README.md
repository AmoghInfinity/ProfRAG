# ProfRAG

A Retrieval-Augmented Generation (RAG) system designed to process and understand documents such as PDFs and PowerPoint files. The system supports intelligent retrieval, reranking, context compression, answer generation, flashcard creation, and evaluation using RAGAS.

---

## Overview

ProfRAG is an end-to-end pipeline that converts raw documents into a queryable knowledge system. It combines retrieval techniques with language models to generate answers grounded in provided context.

The system is modular and designed for experimentation as well as practical usage.

---

## Features

- Document ingestion (PDF, PPT)
- Context-aware chunking
- BGE embeddings for semantic representation
- Hybrid retrieval
- Cross-encoder reranking
- Context compression
- LLM-based answer generation (Groq)
- Flashcard generation and export
- RAGAS-based evaluation (faithfulness, relevancy, precision, recall)
- Streamlit interface

---

## Architecture
```
Raw Documents (PDF / PPT)
        │
        ▼
Text Extraction
        │
        ▼
Chunking
        │
        ▼
Embeddings (BGE)
        │
        ▼
Hybrid Retrieval
        │
        ▼
Reranking
        │
        ▼
Context Compression
        │
        ▼
LLM Generation (Groq)
        │
        ▼
Evaluation (RAGAS)
```
---

## Project Structure
```
ProfRAG/
│── ingestion/
│   ├── pdf_parser.py
│   ├── ppt_parser.py
│
│── chunking/
│   └── context_chunker.py
│
│── embeddings/
│   └── bge_embedder.py
│
│── retrieval/
│   └── hybrid_retriever.py
│
│── reranking/
│   └── reranker.py
│
│── compression/
│   └── context_compressor.py
│
│── generation/
│   └── generator.py
│
│── evaluation/
│   └── ragas_eval.py
│
│── utils/
│
│── data/
│   └── raw/
│
│── app.py
│── main.py
│── requirements.txt
│── .gitignore
│── README.md
```
---

## Installation

```bash
git clone https://github.com/AmoghInfinity/ProfRAG.git
cd ProfRAG

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

---

## Usage

### Run pipeline

```bash
python main.py
```

### Run Streamlit UI

```bash
streamlit run app.py
```

---

## Example Queries

- What is SVM?
- Advantages of SVM
- Types of SVM
- What is a classification algorithm?

---

## Evaluation

The system evaluates performance using:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Example output:

```
faithfulness: 0.82
answer_relevancy: 0.79
context_precision: 0.88
context_recall: 0.85
```

---

## Flashcards

Flashcards can be generated from retrieved content.

Example format:

```
Q: What is SVM?
A: A supervised learning algorithm used for classification and regression.
```

---

## Design Choices

- BGE embeddings for semantic similarity
- Hybrid retrieval for balanced results
- Cross-encoder reranking for improved ranking
- Context compression to reduce irrelevant content
- Groq for fast inference
- RAGAS for evaluation

---

## Future Scope

- Vector database integration (FAISS or similar)
- Query rewriting or expansion
- Improved retrieval strategies
- Evaluation improvements and benchmarking

---

## Author

Amogh Gupta
