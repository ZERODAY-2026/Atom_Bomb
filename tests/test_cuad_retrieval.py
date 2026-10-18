
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore[missing-import]
from huggingface_hub import hf_hub_download
from pdf import find_relevant_passages


def normalize(text):
    """Normalize whitespace and case before comparing text."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    dataset_path = hf_hub_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        filename="CUAD_v1/CUAD_v1.json",
    )

    with open(dataset_path, encoding="utf-8") as file:
        dataset = json.load(file)

    tested = 0
    hits = 0

    print("CUAD RETRIEVAL EVALUATION\n")

    for contract in dataset["data"]:
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]

            for question_data in paragraph["qas"]:
                answers = question_data.get("answers", [])

                if not answers:
                    continue

                expected = answers[0].get("text", "").strip()
                question = question_data["question"]

                if not expected:
                    continue

                # Skip document-title questions in this test.
                if "document name" in question.lower():
                    continue

                pages = [{"page": 1, "text": context}]

                results = find_relevant_passages(
                    pages,
                    question,
                    max_passages=5,
                )

                normalized_expected = normalize(expected)

                found = any(
                    normalized_expected in normalize(result["text"])
                    for result in results
                )

                tested += 1
                hits += int(found)

                print(
                    f"Test {tested}: "
                    f"{'PASS' if found else 'MISS'}"
                )
                print("Question:", question[:160])
                print("Expected answer:", expected[:160])
                print("Found in retrieved text:", found)
                print()

                if tested >= 5:
                    break

            if tested >= 5:
                break

        if tested >= 5:
            break

    print(
        f"SUMMARY: {hits} of {tested} labelled answers "
        "found in the top five passages"
    )


if __name__ == "__main__":
    main()
