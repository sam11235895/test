"""Implementation of a tiny multinomial Naive Bayes classifier for emails."""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from .data_models import EmailMessage, LabeledEmail
from .preprocessing import tokenize_email

Tokeniser = Callable[[EmailMessage], Sequence[str]]


@dataclass
class _EmailStatistics:
    """Internal helper that stores statistics required for prediction."""

    class_log_prior: Dict[str, float]
    feature_log_prob: Dict[str, Dict[str, float]]
    unknown_log_prob: Dict[str, float]
    vocabulary: List[str]


class NaiveBayesEmailClassifier:
    """A minimal multinomial Naive Bayes text classifier.

    The implementation is intentionally dependency free so that the classifier can
    run in lightweight environments (for instance, a serverless function).  It is
    not as feature rich as scikit-learn, but it provides all of the pieces needed
    for this project: training, prediction and persistence.
    """

    def __init__(self, *, alpha: float = 1.0, tokeniser: Tokeniser | None = None) -> None:
        if alpha <= 0:
            raise ValueError("Alpha (Laplace smoothing parameter) must be positive.")
        self.alpha = alpha
        self.tokeniser = tokeniser or tokenize_email
        self._statistics: _EmailStatistics | None = None

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def fit(self, examples: Iterable[LabeledEmail]) -> None:
        """Train the classifier on labeled examples."""

        class_counts: Counter[str] = Counter()
        token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        total_tokens: Counter[str] = Counter()
        vocabulary: set[str] = set()

        example_list = list(examples)
        if not example_list:
            raise ValueError("Training requires at least one labeled email.")

        for example in example_list:
            tokens = list(self.tokeniser(example.email))
            class_counts[example.label] += 1
            total_tokens[example.label] += len(tokens)

            for token in tokens:
                token_counts[example.label][token] += 1
            vocabulary.update(tokens)

        total_examples = sum(class_counts.values())
        class_log_prior = {
            label: math.log(count / total_examples) for label, count in class_counts.items()
        }

        vocab_size = max(len(vocabulary), 1)
        feature_log_prob: Dict[str, Dict[str, float]] = {}
        unknown_log_prob: Dict[str, float] = {}

        for label in class_counts:
            label_token_counts = token_counts[label]
            label_total = total_tokens[label]
            denom = label_total + self.alpha * vocab_size
            feature_log_prob[label] = {}
            for token in vocabulary:
                count = label_token_counts[token]
                prob = (count + self.alpha) / denom
                feature_log_prob[label][token] = math.log(prob)
            unknown_log_prob[label] = math.log(self.alpha / denom)

        self._statistics = _EmailStatistics(
            class_log_prior=class_log_prior,
            feature_log_prob=feature_log_prob,
            unknown_log_prob=unknown_log_prob,
            vocabulary=sorted(vocabulary),
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _require_statistics(self) -> _EmailStatistics:
        if self._statistics is None:
            raise RuntimeError("The classifier has not been trained yet.")
        return self._statistics

    def predict_proba(self, email: EmailMessage) -> Dict[str, float]:
        """Return posterior probabilities for each known class."""

        stats = self._require_statistics()
        tokens = list(self.tokeniser(email))
        scores: Dict[str, float] = {}
        for label, log_prior in stats.class_log_prior.items():
            score = log_prior
            feature_probs = stats.feature_log_prob[label]
            unknown = stats.unknown_log_prob[label]
            for token in tokens:
                score += feature_probs.get(token, unknown)
            scores[label] = score

        # Convert log probabilities back into a normalised distribution.
        max_log = max(scores.values())
        exp_scores = {label: math.exp(log_prob - max_log) for label, log_prob in scores.items()}
        norm = sum(exp_scores.values())
        return {label: value / norm for label, value in exp_scores.items()}

    def predict(self, email: EmailMessage) -> str:
        """Return the most likely label for *email*."""

        probabilities = self.predict_proba(email)
        return max(probabilities, key=probabilities.get)

    def classify_many(self, emails: Iterable[EmailMessage]) -> List[str]:
        """Classify a sequence of emails and return their predicted labels."""

        return [self.predict(email) for email in emails]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise the trained classifier to *path* using pickle."""

        stats = self._require_statistics()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "alpha": self.alpha,
                "statistics": stats,
            }, fh)

    @classmethod
    def load(cls, path: str | Path, *, tokeniser: Tokeniser | None = None) -> "NaiveBayesEmailClassifier":
        """Load a previously saved classifier from *path*."""

        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        classifier = cls(alpha=payload["alpha"], tokeniser=tokeniser)
        classifier._statistics = payload["statistics"]
        return classifier
