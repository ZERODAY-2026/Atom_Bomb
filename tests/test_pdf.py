
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf import extract_pdf, find_relevant_passages

project_root = Path(__file__).resolve().parents[1]
pages = extract_pdf(project_root / "sample.pdf")

question = "How many days notice must a tenant provide?"

passages = find_relevant_passages(pages, question)

print("\nRelevant evidence:")

for passage in passages:
    print(f"\nID: {passage['id']}")
    print(f"Page: {passage['page']}")
    print(f"Text: {passage['text']}")
