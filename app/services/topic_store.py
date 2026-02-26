import json
import os
from typing import Dict, Any


class TopicStore:
    """Load/update topics stored in data/topic_keywords.json"""

    def __init__(self):
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "topic_keywords.json",
        )

    def load_topics(self) -> Dict[str, Dict[str, Any]]:
        with open(self.data_file, "r") as f:
            return json.load(f)

    def save_topics(self, topics: Dict[str, Dict[str, Any]]) -> None:
        with open(self.data_file, "w") as f:
            json.dump(topics, f, indent=2)

    def add_topic(self, name: str, description: str, overwrite: bool = False) -> None:
        topics = self.load_topics()

        if name in topics and not overwrite:
            raise ValueError(
                f"Topic '{name}' already exists. Set overwrite=True to replace."
            )

        topics[name] = {"description": description}
        self.save_topics(topics)