from dataclasses import dataclass

@dataclass
class Email:
    """Dataclass representing an email with subject and body"""
    subject: str
    body: str