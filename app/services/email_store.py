import json
import os
from typing import Any, Dict, List, Optional


class EmailStore:
    """Store/retrieve emails in data/emails.json.

    This class is robust to two formats:
    1) Legacy: a list of emails
    2) Current: {"next_id": int, "emails": [...]}
    """

    def __init__(self):
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "emails.json",
        )

        # Initialize file if missing
        if not os.path.exists(self.data_file):
            self._save({"next_id": 1, "emails": []})

    def _load_raw(self) -> Any:
        with open(self.data_file, "r") as f:
            return json.load(f)

    def _normalize(self, raw: Any) -> Dict[str, Any]:
        # If legacy format is a list, convert it
        if isinstance(raw, list):
            emails = raw
            next_id = 1
            # If legacy items have "id", compute next_id safely
            try:
                ids = [e.get("id", 0) for e in emails if isinstance(e, dict)]
                next_id = (max(ids) + 1) if ids else 1
            except Exception:
                next_id = len(emails) + 1
            return {"next_id": next_id, "emails": emails}

        # If correct dict format, ensure keys exist
        if isinstance(raw, dict):
            if "emails" not in raw or not isinstance(raw["emails"], list):
                raw["emails"] = []
            if "next_id" not in raw or not isinstance(raw["next_id"], int):
                # derive next_id from existing emails
                try:
                    ids = [e.get("id", 0) for e in raw["emails"] if isinstance(e, dict)]
                    raw["next_id"] = (max(ids) + 1) if ids else 1
                except Exception:
                    raw["next_id"] = len(raw["emails"]) + 1
            return raw

        # Fallback: reset
        return {"next_id": 1, "emails": []}

    def _load(self) -> Dict[str, Any]:
        raw = self._load_raw()
        data = self._normalize(raw)

        # If it was legacy, write back normalized format once
        if not isinstance(raw, dict):
            self._save(data)

        return data

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_email(
        self,
        subject: str,
        body: str,
        topic: Optional[str],
        embedding: List[float],
    ) -> int:
        data = self._load()

        email_id = data["next_id"]
        data["next_id"] += 1

        data["emails"].append(
            {
                "id": email_id,
                "subject": subject,
                "body": body,
                "topic": topic,          # optional ground truth
                "embedding": embedding,  # vector for later similarity search
            }
        )

        self._save(data)
        return email_id

    def list_emails(self) -> List[Dict[str, Any]]:
        return self._load()["emails"]