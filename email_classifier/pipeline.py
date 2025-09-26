"""High level helpers for working with the email classifier."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .data_models import EmailMessage, LabeledEmail
from .naive_bayes import NaiveBayesEmailClassifier

Dataset = Sequence[LabeledEmail]


@dataclass
class EvaluationResult:
    """Simple container with accuracy statistics for a classifier run."""

    accuracy: float
    total_examples: int

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "total_examples": self.total_examples}


def load_labeled_dataset(path: str | Path) -> List[LabeledEmail]:
    """Load a JSON file containing training examples.

    The file must contain an array of objects with the following keys::

        {
            "sender": "billing@example.com",
            "subject": "Your invoice",
            "body": "The body text",
            "label": "finance"
        }
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    dataset = []
    for entry in payload:
        dataset.append(
            LabeledEmail(
                email=EmailMessage(
                    sender=entry.get("sender", ""),
                    subject=entry.get("subject", ""),
                    body=entry.get("body", ""),
                ),
                label=entry["label"],
            )
        )
    return dataset


def load_unlabeled_dataset(path: str | Path) -> List[EmailMessage]:
    """Load a JSON file containing raw emails without labels."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    dataset = []
    for entry in payload:
        dataset.append(
            EmailMessage(
                sender=entry.get("sender", ""),
                subject=entry.get("subject", ""),
                body=entry.get("body", ""),
            )
        )
    return dataset


class EmailClassificationPipeline:
    """Thin convenience wrapper around :class:`NaiveBayesEmailClassifier`."""

    def __init__(self, classifier: NaiveBayesEmailClassifier | None = None) -> None:
        self.classifier = classifier or NaiveBayesEmailClassifier()

    def train(self, dataset: Iterable[LabeledEmail]) -> None:
        self.classifier.fit(list(dataset))

    def train_from_path(self, path: str | Path) -> None:
        dataset = load_labeled_dataset(path)
        self.train(dataset)

    def evaluate(self, dataset: Iterable[LabeledEmail]) -> EvaluationResult:
        dataset_list = list(dataset)
        if not dataset_list:
            raise ValueError("The evaluation dataset must contain at least one example.")

        predictions = self.classifier.classify_many([example.email for example in dataset_list])
        correct = sum(
            1 for prediction, example in zip(predictions, dataset_list) if prediction == example.label
        )
        accuracy = correct / len(dataset_list)
        return EvaluationResult(accuracy=accuracy, total_examples=len(dataset_list))

    def train_with_validation(
        self,
        dataset: Sequence[LabeledEmail],
        *,
        validation_split: float = 0.2,
        random_seed: int | None = None,
    ) -> EvaluationResult | None:
        """Train the model and optionally evaluate it on a validation split."""

        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1).")

        dataset_list = list(dataset)
        if not dataset_list:
            raise ValueError("Training requires at least one example.")

        if validation_split == 0 or len(dataset_list) == 1:
            self.train(dataset_list)
            return None

        rng = random.Random(random_seed)
        shuffled = dataset_list[:]
        rng.shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * (1 - validation_split)))
        training_data = shuffled[:split_index]
        validation_data = shuffled[split_index:]

        self.train(training_data)
        return self.evaluate(validation_data)

    def predict(self, email: EmailMessage) -> str:
        return self.classifier.predict(email)

    def predict_many(self, emails: Iterable[EmailMessage]) -> List[str]:
        return self.classifier.classify_many(emails)

    def predict_with_probabilities(self, email: EmailMessage) -> dict[str, float]:
        return self.classifier.predict_proba(email)

    def save(self, path: str | Path) -> None:
        self.classifier.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "EmailClassificationPipeline":
        classifier = NaiveBayesEmailClassifier.load(path)
        return cls(classifier)


def split_dataset(
    dataset: Sequence[LabeledEmail],
    *,
    validation_split: float,
    random_seed: int | None = None,
) -> Tuple[List[LabeledEmail], List[LabeledEmail]]:
    """Split *dataset* into training and validation sets."""

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be strictly between 0 and 1.")

    rng = random.Random(random_seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1 - validation_split)))
    return shuffled[:split_index], shuffled[split_index:]
