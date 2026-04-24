import streamlit as st
import os
import json

from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.ppt_parser import extract_text_from_ppt
from chunking.context_chunker import chunk_documents
from embeddings.bge_embedder import BGEEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import CrossEncoderReranker
from compression.context_compressor import ContextCompressor
from generation.generator import Generator


# ---------------- CONFIG ----------------
st.set_page_config(page_title="ProfRAG", layout="wide")

st.title("📚 ProfRAG")
st.caption("Your AI-powered study assistant")

# ---------------- SESSION STATE ----------------
for key in ["pipeline_ready", "retriever", "embedder", "reranker",
            "compressor", "generator", "chat_history", "file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF / PPT",
    type=["pdf", "pptx"]
)

mode = st.sidebar.radio(
    "Mode",
    ["💬 Q&A", "🧠 Flashcards", "📝 Exam Questions"]
)

if st.sidebar.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()

# ---------------- PIPELINE ----------------
def build_pipeline(file_path):
    if file_path.endswith(".pdf"):
        docs = extract_text_from_pdf(file_path)
    else:
        docs = extract_text_from_ppt(file_path)

    chunks = chunk_documents(docs)

    embedder = BGEEmbedder()
    embedded_docs = embedder.embed_documents(chunks)

    retriever = HybridRetriever()
    retriever.build_index(embedded_docs)

    reranker = CrossEncoderReranker()
    compressor = ContextCompressor()
    generator = Generator()

    return retriever, embedder, reranker, compressor, generator


# ---------------- FILE HANDLING (FIXED) ----------------
UPLOAD_DIR = "data/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if uploaded_file:

    if not st.session_state.pipeline_ready:
        with st.spinner("⚙️ Processing your file..."):

            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            if not os.path.exists(file_path):
                st.error("File saving failed!")
                st.stop()

            retriever, embedder, reranker, compressor, generator = build_pipeline(file_path)

            st.session_state.retriever = retriever
            st.session_state.embedder = embedder
            st.session_state.reranker = reranker
            st.session_state.compressor = compressor
            st.session_state.generator = generator
            st.session_state.pipeline_ready = True
            st.session_state.file_name = uploaded_file.name

        st.success(f"✅ Loaded: {uploaded_file.name}")

# ---------------- MAIN UI ----------------
if st.session_state.pipeline_ready:

    st.markdown(f"📄 **Current File:** `{st.session_state.file_name}`")

    # Chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Input
    query = st.chat_input("Ask something...")

    if query:

        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("🧠 Thinking..."):

                retriever = st.session_state.retriever
                embedder = st.session_state.embedder
                reranker = st.session_state.reranker
                compressor = st.session_state.compressor
                generator = st.session_state.generator

                if mode == "💬 Q&A":

                    results = retriever.retrieve(query, embedder, top_k=10)
                    results = reranker.rerank(query, results)
                    results = compressor.compress(query, results)

                    output = generator.generate_answer(query, results)

                elif mode == "🧠 Flashcards":

                    results = retriever.retrieve("important concepts", embedder, top_k=10)
                    results = reranker.rerank("important concepts", results)
                    results = compressor.compress("important concepts", results)

                    output = generator.generate_flashcards(results)

                else:  # Exam Questions

                    results = retriever.retrieve("important topics", embedder, top_k=10)
                    results = reranker.rerank("important topics", results)
                    results = compressor.compress("important topics", results)

                    output = generator.generate_exam_questions(results)

                st.markdown(output)

                # Save response
                st.session_state.chat_history.append({"role": "assistant", "content": output})

                # ---------------- DOWNLOAD ----------------
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📥 Download Output",
                        data=output,
                        file_name="output.txt"
                    )

                with col2:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps({"output": output}, indent=2),
                        file_name="output.json"
                    )

                # ---------------- SOURCES ----------------
                with st.expander("📌 Sources"):
                    for r in results[:3]:
                        st.write(r["metadata"])

else:
    st.info("👈 Upload a file to begin")