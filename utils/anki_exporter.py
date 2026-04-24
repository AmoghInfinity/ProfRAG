import json
import csv
import re


def parse_flashcards(raw_output: str):
    """
    Robust JSON parser for LLM output
    """

    try:
        return json.loads(raw_output)
    except:
        # 🔥 Try to fix broken JSON
        try:
            cleaned = re.search(r"\[.*\]", raw_output, re.DOTALL).group()
            return json.loads(cleaned)
        except:
            return []


def export_to_csv(flashcards, file_path="flashcards.csv"):
    """
    Export ALL valid flashcards to Anki CSV
    """

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ✅ REQUIRED for Anki
        writer.writerow(["Front", "Back"])

        count = 0

        for card in flashcards:
            q = card.get("question", "").strip()
            a = card.get("answer", "").strip()

            # Skip garbage safely (not aggressively)
            if not q or not a:
                continue

            writer.writerow([q, a])
            count += 1

        print(f"Exported {count} flashcards")

    return file_path