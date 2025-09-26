"""Data models for representing email messages and labeled training examples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EmailMessage:
    """Container for the pieces of an email that we care about.

    The classifier intentionally keeps the representation simple: sender,
    subject and body text.  Additional metadata (for example the recipients or
    attachments) can be added in the future without changing the overall
    structure.
    """

    sender: str
    subject: str
    body: str

    def as_text(self) -> str:
        """Return a single text blob containing all relevant email content."""

        parts = [self.sender or "", self.subject or "", self.body or ""]
        return "\n".join(part.strip() for part in parts if part is not None)


@dataclass(frozen=True)
class LabeledEmail:
    """A training example composed of an :class:`EmailMessage` and a label."""

    email: EmailMessage
    label: str

    def to_dict(self) -> Dict[str, str]:
        """Serialize the labeled email to a JSON friendly dictionary."""

        return {
            "sender": self.email.sender,
            "subject": self.email.subject,
            "body": self.email.body,
            "label": self.label,
        }
