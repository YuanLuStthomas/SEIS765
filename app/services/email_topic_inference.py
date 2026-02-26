from typing import Dict, Any
import numpy as np

from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email
from app.services.email_store import EmailStore


class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""

    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()

    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""

        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)

        # Step 2: Classify using features
        predicted_topic = self.model.predict(features)
        topic_scores = self.model.get_topic_scores(features)

        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email,
        }

    def classify_email_by_stored_emails(self, email: Email) -> Dict[str, Any]:
        """
        Predict topic by finding the most similar stored email (with ground truth topic).
        This uses cosine similarity between embeddings.
        """

        # Only need embeddings for this mode
        features = self.feature_factory.generate_all_features(
            email, generator_names=["email_embeddings"]
        )
        query_emb = features.get("email_embeddings_average_embedding", None)
        if query_emb is None:
            raise ValueError("Missing email embedding feature for email-based inference.")

        if isinstance(query_emb, list):
            query_emb = np.array(query_emb, dtype=float)

        store = EmailStore()
        stored = store.list_emails()

        # Only compare against stored emails that have a ground truth topic
        labeled = [e for e in stored if e.get("topic")]
        if not labeled:
            raise ValueError("No stored emails with ground truth topic available.")

        best_email = None
        best_score = -1.0

        for e in labeled:
            emb = e.get("embedding", None)
            if emb is None:
                continue

            if isinstance(emb, list):
                emb = np.array(emb, dtype=float)

            denom = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            score = float(np.dot(query_emb, emb) / denom) if denom != 0 else -1.0

            if score > best_score:
                best_score = score
                best_email = e

        if best_email is None:
            raise ValueError("Could not compute similarity against stored emails.")

        predicted = best_email["topic"]

        return {
            "predicted_topic": predicted,
            "topic_scores": {predicted: best_score},
            "features": features,
            "available_topics": self.model.topics,
            "nearest_email_id": best_email.get("id"),
            "nearest_email_score": best_score,
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions(),
        }