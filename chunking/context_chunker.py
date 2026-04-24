from typing import List, Dict
import re


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and clean text.
    """
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter using regex.
    """
    text = normalize_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_sentence(sentence: str, max_words: int) -> List[str]:
    """
    Handle edge case where a single sentence exceeds max_words.
    """
    words = sentence.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def create_chunks_from_sentences(sentences: List[str], max_words: int = 200) -> List[str]:
    """
    Combine sentences into chunks with a word limit.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # Handle very long sentence
        if word_count > max_words:
            long_chunks = split_long_sentence(sentence, max_words)
            chunks.extend(long_chunks)
            continue

        # If adding sentence exceeds limit → push current chunk
        if current_length + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_documents(documents: List[Dict], max_words: int = 200) -> List[Dict]:
    """
    Chunk documents while preserving metadata (LangChain-ready format)
    """
    chunked_docs = []

    for doc in documents:
        text = normalize_text(doc.get("text", ""))
        metadata = doc.get("metadata", {})

        if not text:
            continue

        sentences = split_into_sentences(text)
        chunks = create_chunks_from_sentences(sentences, max_words)

        for idx, chunk in enumerate(chunks):
            new_metadata = metadata.copy()

            # Only extend metadata (no overwriting)
            new_metadata["chunk_id"] = idx

            chunked_docs.append({
                "text": chunk,
                "metadata": new_metadata
            })

    return chunked_docs


# ------------------ REAL PIPELINE TEST ------------------

if __name__ == "__main__":
    from ingestion.pdf_parser import extract_text_from_pdf
    from ingestion.ppt_parser import extract_text_from_ppt

    pdf_path = "data/raw/sample.pdf"
    ppt_path = "data/raw/sample.pptx"

    print("\n🔹 Testing PDF → Chunking Pipeline\n")
    pdf_docs = extract_text_from_pdf(pdf_path)
    pdf_chunks = chunk_documents(pdf_docs)

    for chunk in pdf_chunks[:3]:
        print("\n--- PDF Chunk ---")
        print(chunk["text"][:200])
        print(chunk["metadata"])

    print("\n🔹 Testing PPT → Chunking Pipeline\n")
    ppt_docs = extract_text_from_ppt(ppt_path)
    ppt_chunks = chunk_documents(ppt_docs)

    for chunk in ppt_chunks[:3]:
        print("\n--- PPT Chunk ---")
        print(chunk["text"][:200])
        print(chunk["metadata"])

