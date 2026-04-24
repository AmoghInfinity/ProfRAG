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
            model="llama-3.1-8b-instant",  # ✅ stable model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # 🔥 lower = more factual
        )

        return response.choices[0].message.content.strip()

    def generate_flashcards(self, documents: List[Dict]) -> str:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
Create revision flashcards.

Rules:
- Short answers
- No repetition
- Only from context

Format:

Q: ...
A: ...

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
Generate 5 exam questions.

Rules:
- Conceptual + theoretical
- No repetition
- Only from context

Content:
{context}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()