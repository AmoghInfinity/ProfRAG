import os
import re
from typing import List, Dict
from collections import Counter
from pypdf import PdfReader


def extract_raw_text(reader) -> List[List[str]]:
    """
    Extract raw lines from each page.
    """
    pages_lines = []

    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        pages_lines.append(lines)

    return pages_lines


def detect_repeated_lines(pages_lines: List[List[str]], threshold: float = 0.6):
    """
    Detect lines that appear in many pages (headers/footers).
    """
    line_counter = Counter()

    for lines in pages_lines:
        unique_lines = set(lines)
        line_counter.update(unique_lines)

    total_pages = len(pages_lines)

    repeated = {
        line for line, count in line_counter.items()
        if count / total_pages >= threshold
    }

    return repeated


def is_noise_line(line: str) -> bool:
    import re

    line = line.strip()

    if not line:
        return True

    # Pure numbers
    if re.fullmatch(r"\d+", line):
        return True

    # High digit ratio
    digits = sum(c.isdigit() for c in line)
    if len(line) > 0 and digits / len(line) > 0.3:
        return True

    # Contains academic header patterns
    if re.search(r"(module|lecture|chapter|unit)", line, re.IGNORECASE):
        return True

    # Contains mostly uppercase short text (headers)
    words = line.split()
    if len(words) <= 5 and all(w.isupper() for w in words if w.isalpha()):
        return True

    # Very short
    if len(line) < 3:
        return True

    return False


def clean_pages(pages_lines: List[List[str]], repeated_lines: set) -> List[List[str]]:
    """
    Clean pages by removing repeated lines and noise.
    """
    cleaned_pages = []

    for lines in pages_lines:
        cleaned = []

        for line in lines:
            if line in repeated_lines:
                continue

            if is_noise_line(line):
                continue

            cleaned.append(line)

        cleaned_pages.append(cleaned)

    return cleaned_pages


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Final output in LangChain-friendly format
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    file_name = os.path.basename(file_path)

    # Step 1: Extract raw lines
    pages_lines = extract_raw_text(reader)

    # Step 2: Detect repeated lines
    repeated_lines = detect_repeated_lines(pages_lines)

    # Step 3: Clean pages
    cleaned_pages = clean_pages(pages_lines, repeated_lines)

    documents = []

    for i, lines in enumerate(cleaned_pages):
        text = " ".join(lines).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": "pdf",
                "file_name": file_name,
                "page": i + 1
            }
        })

    return documents


if __name__ == "__main__":
    sample_path = "data/raw/sample.pdf"
    docs = extract_text_from_pdf(sample_path)

    for doc in docs[:2]:
        print("\n--- Document ---")
        print(doc["text"][:300])
        print(doc["metadata"])