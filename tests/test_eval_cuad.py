import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_cuad  # pyright: ignore[reportMissingImports]

CONTRACT = (
    "DISTRIBUTION AGREEMENT\n"
    "This Agreement is governed by the laws of the State of California.\n"
    "The Distributor shall pay all invoices within 30 days of receipt.\n"
    "Either party may terminate this Agreement with 60 days written notice.\n"
)

DATASET = {"data": [{"title": "T", "paragraphs": [{"context": CONTRACT, "qas": [
    {"id": "T__Governing Law",
     "question": 'Highlight the parts related to "Governing Law" that should be reviewed by a lawyer. Details: Which state or country\'s law governs the contract?',
     "answers": [{"text": "governed by the laws of the State of California", "answer_start": 0}]},
    {"id": "T__Notice Period To Terminate Renewal",
     "question": 'Highlight the parts related to "Termination" that should be reviewed by a lawyer. Details: notice to terminate',
     "answers": [{"text": "terminate this Agreement with 60 days written notice", "answer_start": 0}]},
    {"id": "T__Non-Compete",
     "question": 'Highlight the parts related to "Non-Compete" that should be reviewed by a lawyer. Details: restrictions on competing',
     "answers": []},
]}]}]}


def test_evaluate_finds_labelled_clauses():
    result = eval_cuad.evaluate(DATASET, top_k=5)
    assert result["labelled_questions"] == 2
    assert result["hits"] == 2
    assert result["hit_rate"] == 1.0
    assert result["unanswerable_questions"] == 1


def test_is_hit_rejects_unrelated_passage():
    passages = [{"text": "The weather is nice and the sky is blue today."}]
    answers = [{"text": "governed by the laws of the State of California"}]
    assert eval_cuad.is_hit(passages, answers) is False
