"""
Offline retrieval evaluation on CUAD (SQuAD-format JSON).

Question it answers: for clauses that CUAD's expert lawyers labelled, does
find_relevant_passages() bring the labelled text back in its top-k results?

Setup (one-time, NOT part of normal app use):
  1. Download data.zip from https://github.com/TheAtticusProject/cuad
  2. Unzip it into ./data/   (CUADv1.json, test.json, ...)  -- git-ignored
Run:
  python scripts/eval_cuad.py data/test.json --max-contracts 50
Results are printed and written to results/cuad_eval.json.

CAVEAT: CUAD is commercial contracts. These numbers measure clause retrieval
on that kind of document only; they do not prove accuracy on rental or
employment agreements.
Data: CUAD, CC BY 4.0 (Hendrycks et al., NeurIPS 2021).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf import find_relevant_passages, normalize_for_match  # noqa: E402

PAGE_CHARS = 3000


def split_into_pages(text, size=PAGE_CHARS):
    """CUAD contracts are plain text; cut them into pseudo-pages on line breaks."""
    pages, current, count = [], [], 0
    for line in text.splitlines():
        current.append(line)
        count += len(line) + 1
        if count >= size:
            pages.append({"page": len(pages) + 1, "text": "\n".join(current)})
            current, count = [], 0
    if any(l.strip() for l in current):
        pages.append({"page": len(pages) + 1, "text": "\n".join(current)})
    return pages


def clause_type(qa):
    qid = qa.get("id", "")
    if "__" in qid:
        return qid.split("__")[-1]
    match = re.search(r'"([^"]+)"', qa.get("question", ""))
    return match.group(1) if match else "unknown"


def _words(text):
    return set(re.findall(r"[a-z0-9]+", normalize_for_match(text)))


def is_hit(passages, answers, threshold=0.5):
    """A passage matches if it contains the labelled text, or shares >= 50% of its words."""
    for answer in answers:
        gold = normalize_for_match(answer.get("text", ""))
        if not gold:
            continue
        gold_words = _words(gold)
        for passage in passages:
            found = normalize_for_match(passage["text"])
            if gold in found or (found and found in gold):
                return True
            if gold_words and len(gold_words & _words(found)) / len(gold_words) >= threshold:
                return True
    return False


def evaluate(dataset, top_k=5, max_contracts=None, progress_callback=None):
    """Evaluate retrieval on a CUAD format dataset."""
    per_type = defaultdict(lambda: {"asked": 0, "hits": 0})
    unanswerable = {"asked": 0, "returned_something": 0}
    contracts = 0

    contract_list = dataset.get("data", [])
    total_to_process = min(len(contract_list), max_contracts) if max_contracts is not None else len(contract_list)

    for contract in contract_list:
        if max_contracts is not None and contracts >= max_contracts:
            break
        contracts += 1
        if progress_callback:
            progress_callback(contracts, total_to_process)

        for paragraph in contract.get("paragraphs", []):
            pages = split_into_pages(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                passages = find_relevant_passages(pages, qa.get("question", ""), max_passages=top_k)
                if answers:
                    row = per_type[clause_type(qa)]
                    row["asked"] += 1
                    row["hits"] += int(is_hit(passages, answers))
                else:
                    unanswerable["asked"] += 1
                    unanswerable["returned_something"] += int(bool(passages))

    asked = sum(r["asked"] for r in per_type.values())
    hits = sum(r["hits"] for r in per_type.values())
    return {
        "top_k": top_k,
        "contracts": contracts,
        "labelled_questions": asked,
        "hits": hits,
        "hit_rate": round(hits / asked, 3) if asked else 0.0,
        "unanswerable_questions": unanswerable["asked"],
        "unanswerable_with_results": unanswerable["returned_something"],
        "by_clause_type": {
            k: {**v, "hit_rate": round(v["hits"] / v["asked"], 3) if v["asked"] else 0.0}
            for k, v in sorted(per_type.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-contracts", type=int, default=50)
    parser.add_argument("--out", default="results/cuad_eval.json")
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as handle:
        dataset = json.load(handle)

    result = evaluate(dataset, args.top_k, args.max_contracts)

    print(f"\nCUAD retrieval evaluation ({result['contracts']} contracts, top-{args.top_k})")
    print(f"Labelled questions: {result['labelled_questions']}   "
          f"Hits: {result['hits']}   Hit rate: {result['hit_rate']}")
    print(f"Questions with no labelled answer that still returned passages: "
          f"{result['unanswerable_with_results']}/{result['unanswerable_questions']}\n")
    print(f"{'Clause type':45s} {'asked':>6s} {'hit rate':>9s}")
    for name, row in sorted(result["by_clause_type"].items(), key=lambda kv: kv[1]["hit_rate"]):
        print(f"{name[:45]:45s} {row['asked']:6d} {row['hit_rate']:9.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
