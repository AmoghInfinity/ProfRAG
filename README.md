# 📚 ProfRAG — AI Study Assistant

ProfRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to help students interact with their study materials (PDFs, PPTs) and generate structured learning outputs such as explanations, flashcards, and exam questions.

---

## 🚀 Features

- 📄 Upload PDFs and PPTs
- 🔍 Ask questions from your notes
- 🧠 Generate structured answers (Definition, Explanation, Key Points)
- 🧾 Create flashcards for revision
- 📝 Generate exam-style questions
- ⚡ Fast hybrid retrieval (Vector + BM25)
- 🎯 Cross-encoder reranking
- 🧹 Context compression for better accuracy
- 💬 Streamlit-based interactive UI

---

## 🧠 Architecture

Ingestion → Chunking → Embeddings → Hybrid Retrieval → Reranking → Compression → Generation

---

## 📁 Project Structure
'''
PROFRAG/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .env
│
├── data/
├── ingestion/
├── chunking/
├── embeddings/
├── retrieval/
├── reranking/
├── compression/
├── generation/
├── utils/
'''
---

## ⚙️ Installation

git clone https://github.com/AmoghInfinity/ProfRAG.git
cd ProfRAG

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

---

## 🔑 Setup API Key

Create .env file:

GROQ_API_KEY=your_api_key_here

---

## ▶️ Run

streamlit run app.py

---

## 📌 Future Improvements

- Anki export
- RAGAS evaluation
- Multi-file support
- Deployment

---
