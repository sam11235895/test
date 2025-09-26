#!/usr/bin/env python3
"""Train the email classifier on a labeled dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from email_classifier import EmailClassificationPipeline, load_labeled_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help="Path to a JSON file containing labeled training examples.",
    )
    parser.add_argument(
        "output",
        help="Where to store the trained model (a pickle file).",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the dataset to reserve for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when shuffling data before splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    dataset = load_labeled_dataset(dataset_path)
    if not dataset:
        raise SystemExit("The training dataset is empty.")

    pipeline = EmailClassificationPipeline()

    validation_split = args.validation_split
    if len(dataset) < 2:
        validation_split = 0
    elif not 0 <= validation_split < 1:
        raise SystemExit("validation-split must be in the range [0, 1).")

    result = pipeline.train_with_validation(
        dataset,
        validation_split=validation_split,
        random_seed=args.seed,
    )

    pipeline.save(output_path)

    if result is None:
        print("Model trained without a validation split (not enough data or disabled).")
    else:
        print(
            f"Validation accuracy: {result.accuracy:.2%} across {result.total_examples} held-out emails."
        )
    print(f"Model stored at {output_path.resolve()}")


if __name__ == "__main__":
    main()
