#!/usr/bin/env python3
"""Classify raw emails using a trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from email_classifier import EmailClassificationPipeline, load_unlabeled_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to a trained model produced by train_model.py")
    parser.add_argument(
        "emails",
        help="JSON file containing emails to classify (see data/sample_inbox.json).",
    )
    parser.add_argument(
        "--output",
        help="Optional path where the JSON classification report will be stored.",
    )
    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        help="Include probabilities for each class when printing results to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = EmailClassificationPipeline.load(Path(args.model))
    emails = load_unlabeled_dataset(Path(args.emails))

    if not emails:
        raise SystemExit("No emails to classify.")

    results = []
    for email in emails:
        prediction = pipeline.predict(email)
        entry = {
            "sender": email.sender,
            "subject": email.subject,
            "body": email.body,
            "predicted_label": prediction,
        }
        if args.show_probabilities:
            entry["probabilities"] = pipeline.predict_with_probabilities(email)
        results.append(entry)

    for result in results:
        subject = result["subject"] or "(no subject)"
        print(f"{subject} -> {result['predicted_label']}")
        if args.show_probabilities:
            probs = result["probabilities"]
            formatted = ", ".join(f"{label}: {value:.2%}" for label, value in sorted(probs.items()))
            print(f"    {formatted}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Results stored at {output_path.resolve()}")


if __name__ == "__main__":
    main()
