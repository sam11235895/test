"""Utilities for training and running an email classification workflow."""
from .data_models import EmailMessage, LabeledEmail
from .naive_bayes import NaiveBayesEmailClassifier
from .pipeline import (
    EmailClassificationPipeline,
    EvaluationResult,
    load_labeled_dataset,
    load_unlabeled_dataset,
    split_dataset,
)

__all__ = [
    "EmailClassificationPipeline",
    "EmailMessage",
    "EvaluationResult",
    "LabeledEmail",
    "NaiveBayesEmailClassifier",
    "load_labeled_dataset",
    "load_unlabeled_dataset",
    "split_dataset",
]
