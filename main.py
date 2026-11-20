"""
main.py - CLI Interface for Modern History & Archival Chronicle Assistant
Project 9: Modern History & Archival Chronicle Assistant (GenAI Midsem Project)
"""

import os
import sys
import json
from rag_engine import answer

PDF_PATH = os.path.join("data", "indian_railways_and_telegraph_history_50p.pdf")

ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What exact annual guaranteed interest rate, expressed as a percentage, did the colonial administration promise private British railway companies under the 1849 contract terms?",
        "expected_status": "FOUND",
        "expected_source": "page_13_chunk_1"
    },
    {
        "id": "Q2",
        "question": "Which engineering obstacle on the Bhor Ghat incline made the reversing station described in Chapter 3 necessary, and what was the gradient ratio?",
        "expected_status": "FOUND",
        "expected_source": "page_26_chunk_1"
    },
    {
        "id": "Q3",
        "question": "What is the ticket price for an Executive Chair Car seat on the Vande Bharat Express between New Delhi and Varanasi?",
        "expected_status": "NOT_FOUND",
        "expected_source": []
    },
    {
        "id": "Q4",
        "question": "According to the archival records in the text, who supervised the laying of the experimental Diamond Harbour underwater telegraph cable in 1851?",
        "expected_status": "FOUND",
        "expected_source": "page_43_chunk_1"
    },
    {
        "id": "Q5",
        "question": "Which country won the 2022 FIFA World Cup?",
        "expected_status": "NOT_FOUND",
        "expected_source": []
    }
]


def check_prerequisites():
    """Validates that necessary files and directories exist before running."""
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] Required knowledge source not found at: '{PDF_PATH}'")
        print("Please place 'indian_railways_and_telegraph_history_50p.pdf' into the 'data/' folder.")
        sys.exit(1)


def run_benchmark():
    """Runs the 5 assessment benchmark questions and prints exact dictionary results."""
    check_prerequisites()

    print("=" * 80)
    print("PROJECT 9: MODERN HISTORY & ARCHIVAL CHRONICLE ASSISTANT")
    print("OFFICIAL ASSESSMENT BENCHMARK VERIFICATION (Q1 - Q5)")
    print(f"Corpus: {PDF_PATH}")
    print("=" * 80 + "\n")

    results = []
    all_passed = True

    for item in ASSESSMENT_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        expected_status = item["expected_status"]

        print(f"--------------------------------------------------------------------------------")
        print(f"[{qid}] QUESTION:")
        print(f"{q}\n")

        res = answer(q)
        ans = res["answer"]
        sources = res["sources"]

        # Exact dictionary output display
        print("EXACT DICTIONARY OUTPUT:")
        print(json.dumps(res, indent=2))
        print()

        # Integrity verification
        if expected_status == "FOUND":
            passed = (ans != "[not found]" and len(sources) > 0 and item["expected_source"] in sources)
        else:
            passed = (ans == "[not found]" and len(sources) == 0)

        if passed:
            print(f"STATUS: PASS (Expected: {expected_status})")
        else:
            print(f"STATUS: FAIL (Expected: {expected_status})")
            all_passed = False

        print(f"--------------------------------------------------------------------------------\n")
        results.append((qid, q, res, passed))

    print("=" * 80)
    passed_count = sum(1 for _, _, _, p in results if p)
    total_count = len(results)
    print(f"OVERALL BENCHMARK RESULT: {passed_count}/{total_count} PASSED ({(passed_count/total_count)*100:.1f}%)")
    print("=" * 80)


def run_interactive():
    check_prerequisites()
    print("=" * 75)
    print("Project 9: Modern History & Archival Chronicle Assistant")
    print(f"Knowledge Source: {PDF_PATH}")
    print("Type your question below, or type 'exit' or 'quit' to stop.")
    print("=" * 75 + "\n")

    while True:
        try:
            query = input("Ask Archival Assistant > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting Archival Chronicle Assistant. Goodbye!")
                break

            result = answer(query)
            print("\n" + json.dumps(result, indent=2) + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--benchmark", "--eval", "--test"]:
            run_benchmark()
        else:
            check_prerequisites()
            q = " ".join(sys.argv[1:])
            res = answer(q)
            print(json.dumps(res, indent=2))
    else:
        run_interactive()
