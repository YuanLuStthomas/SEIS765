from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.services.email_topic_inference import EmailTopicInferenceService
from app.services.topic_store import TopicStore
from app.services.email_store import EmailStore
from app.dataclasses import Email
from app.features.factory import FeatureGeneratorFactory

router = APIRouter()


class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: str = "topic"  # "topic" or "email"


class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str


class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    topic: Optional[str] = None


class TopicAddRequest(BaseModel):
    name: str
    description: str
    overwrite: bool = False


class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]


class EmailAddResponse(BaseModel):
    message: str
    email_id: int


@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)

        if request.mode == "email":
            result = inference_service.classify_email_by_stored_emails(email)
        else:
            result = inference_service.classify_email(email)

        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails", response_model=EmailAddResponse)
async def store_email(request: EmailStoreRequest):
    """Store an email (optionally with ground truth topic) and persist to data/emails.json"""
    try:
        email = Email(subject=request.subject, body=request.body)

        factory = FeatureGeneratorFactory()
        features = factory.generate_all_features(email, generator_names=["email_embeddings"])
        embedding = features["email_embeddings_average_embedding"]

        store = EmailStore()
        email_id = store.add_email(
            subject=request.subject,
            body=request.body,
            topic=request.topic,
            embedding=embedding,
        )

        return EmailAddResponse(message="Email stored", email_id=email_id)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing expected feature key: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def topics():
    """Get available email topics"""
    try:
        inference_service = EmailTopicInferenceService()
        info = inference_service.get_pipeline_info()
        return {"topics": info["available_topics"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics")
async def add_topic(request: TopicAddRequest):
    """Dynamically add a new topic and persist to data/topic_keywords.json"""
    try:
        store = TopicStore()
        store.add_topic(request.name, request.description, overwrite=request.overwrite)

        inference_service = EmailTopicInferenceService()
        inference_service.model.reload_topics()

        return {"message": "Topic added", "topic": request.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info")
async def pipeline_info():
    try:
        inference_service = EmailTopicInferenceService()
        return inference_service.get_pipeline_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features")
async def features():
    """Get available feature generators and their feature names"""
    try:
        factory = FeatureGeneratorFactory()
        return {"available_generators": factory.get_available_generators()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))