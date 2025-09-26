from __future__ import annotations

from pathlib import Path

import pytest

from email_classifier.data_models import EmailMessage, LabeledEmail
from email_classifier.naive_bayes import NaiveBayesEmailClassifier


def make_example(sender: str, subject: str, body: str, label: str) -> LabeledEmail:
    return LabeledEmail(email=EmailMessage(sender=sender, subject=subject, body=body), label=label)


def test_classifier_learns_keyword_associations(tmp_path: Path) -> None:
    dataset = [
        make_example("hr@company.com", "performance review", "schedule review", "work"),
        make_example("finance@bank.com", "statement", "monthly statement ready", "finance"),
        make_example("promotions@shop.com", "discount", "limited offer", "promotions"),
    ]

    classifier = NaiveBayesEmailClassifier(alpha=1.0)
    classifier.fit(dataset)

    work_email = EmailMessage(
        sender="ceo@company.com",
        subject="team meeting",
        body="please prepare the quarterly report",
    )
    finance_email = EmailMessage(
        sender="alerts@bank.com",
        subject="payment received",
        body="your card statement is available",
    )

    assert classifier.predict(work_email) == "work"
    assert classifier.predict(finance_email) == "finance"


def test_probabilities_are_normalised() -> None:
    dataset = [
        make_example("spam@unknown.com", "win", "win big", "spam"),
        make_example("news@site.com", "update", "daily update", "news"),
    ]

    classifier = NaiveBayesEmailClassifier()
    classifier.fit(dataset)

    email = EmailMessage(sender="newsletter@site.com", subject="daily update", body="hello")
    probabilities = classifier.predict_proba(email)

    assert pytest.approx(sum(probabilities.values()), rel=1e-6, abs=1e-6) == 1.0
    assert set(probabilities) == {"spam", "news"}


def test_model_can_be_saved_and_loaded(tmp_path: Path) -> None:
    dataset = [
        make_example("spam@unknown.com", "prize", "click now", "spam"),
        make_example("friend@example.com", "dinner", "catch up soon", "personal"),
    ]

    classifier = NaiveBayesEmailClassifier()
    classifier.fit(dataset)

    model_path = tmp_path / "model.pkl"
    classifier.save(model_path)

    restored = NaiveBayesEmailClassifier.load(model_path)
    email = EmailMessage(sender="friend@example.com", subject="dinner", body="catch up soon")
    assert restored.predict(email) == "personal"
