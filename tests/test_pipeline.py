from __future__ import annotations

import json
from pathlib import Path

import pytest

from email_classifier.data_models import EmailMessage, LabeledEmail
from email_classifier.pipeline import (
    EmailClassificationPipeline,
    EvaluationResult,
    load_labeled_dataset,
    load_unlabeled_dataset,
    split_dataset,
)


def write_dataset(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh)


def test_load_labeled_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        [
            {
                "sender": "a@example.com",
                "subject": "Hello",
                "body": "Body",
                "label": "greeting",
            },
            {
                "sender": "b@example.com",
                "subject": "Invoice",
                "body": "Your invoice",
                "label": "finance",
            },
        ],
    )

    dataset = load_labeled_dataset(dataset_path)
    assert len(dataset) == 2
    assert dataset[0].label == "greeting"
    assert dataset[1].email.subject == "Invoice"


def test_load_unlabeled_dataset(tmp_path: Path) -> None:
    inbox_path = tmp_path / "inbox.json"
    write_dataset(
        inbox_path,
        [
            {"sender": "c@example.com", "subject": "Update", "body": "News"},
            {"sender": "d@example.com", "subject": "Meeting", "body": "Agenda"},
        ],
    )

    inbox = load_unlabeled_dataset(inbox_path)
    assert len(inbox) == 2
    assert inbox[0].subject == "Update"


def test_pipeline_training_and_evaluation() -> None:
    dataset = [
        LabeledEmail(EmailMessage("boss@company.com", "meeting", "report"), "work"),
        LabeledEmail(EmailMessage("hr@company.com", "policy", "update"), "work"),
        LabeledEmail(EmailMessage("sales@shop.com", "discount", "special offer"), "promotions"),
        LabeledEmail(EmailMessage("sales@shop.com", "coupon", "save big"), "promotions"),
    ]

    pipeline = EmailClassificationPipeline()
    result = pipeline.train_with_validation(dataset, validation_split=0.5, random_seed=1)

    assert isinstance(result, EvaluationResult)
    assert 0 <= result.accuracy <= 1
    assert result.total_examples == len(dataset) - len(dataset) // 2

    email = EmailMessage("boss@company.com", "meeting", "report")
    assert pipeline.predict(email) == "work"


def test_split_dataset_errors_when_split_invalid() -> None:
    dataset = [
        LabeledEmail(EmailMessage("a", "b", "c"), "label"),
        LabeledEmail(EmailMessage("d", "e", "f"), "label"),
    ]

    with pytest.raises(ValueError):
        split_dataset(dataset, validation_split=0)
    with pytest.raises(ValueError):
        split_dataset(dataset, validation_split=1)


def test_split_dataset_returns_expected_sizes() -> None:
    dataset = [
        LabeledEmail(EmailMessage(f"sender{i}", f"subject{i}", f"body{i}"), "label")
        for i in range(10)
    ]

    train, validation = split_dataset(dataset, validation_split=0.3, random_seed=42)
    assert len(train) + len(validation) == len(dataset)
    assert len(validation) >= 1
