from typing import List, Dict
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


class Generator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)

    def generate_answer(self, query: str, documents: List[Dict]) -> str:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are a highly accurate AI tutor.

STRICT RULES:
- Use ONLY the given context
- Do NOT add external knowledge
- Avoid repetition
- Be clear and structured

Format:

Definition:
(only if present)

Explanation:
(simple and clear)

Key Points:
- bullet points
- no repetition

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    # 🔥 FINAL FLASHCARDS (STRONG + COMPLETE)
    def generate_flashcards(self, documents: List[Dict]) -> str:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are generating high-quality study flashcards.

STRICT RULES:
- Every flashcard MUST be a proper question
- Convert concepts into full questions
- Answers must be COMPLETE but concise (1–2 lines)
- No repetition
- Use ONLY the provided content
- Generate at least 8–12 flashcards if possible

Return ONLY valid JSON:

[
  {{
    "question": "What is ...?",
    "answer": "Clear and complete answer..."
  }}
]

Content:
{context}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def generate_exam_questions(self, documents: List[Dict]) -> str:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are an exam setter.

Generate 5 high-quality exam questions.

STRICT RULES:
- Mix conceptual + theoretical questions
- No repetition
- Use ONLY the provided content

Content:
{context}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()