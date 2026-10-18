
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import ask_gemma


def main():
    passages = [
        {
            "id": "P1",
            "page": 2,
            "text": (
                "The tenant must provide 30 days' written notice "
                "before ending the tenancy."
            ),
        },
        {
            "id": "P2",
            "page": 3,
            "text": (
                "The security deposit will be returned within "
                "15 days after the tenant vacates, subject to "
                "deductions permitted by the agreement."
            ),
        },
    ]

    question = "How much notice must the tenant give?"

    print("Question:", question)
    print("\nGemma's answer:\n")

    answer = ask_gemma(question, passages)
    print(answer)


if __name__ == "__main__":
    main()
