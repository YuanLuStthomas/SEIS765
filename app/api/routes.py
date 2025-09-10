from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

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
        result = inference_service.classify_email(email)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: Students will implement this endpoint
# @router.post("/emails", response_model=EmailAddResponse)
# async def add_email(request: EmailWithTopicRequest):
#     """Add an email with its ground truth topic to the database"""
#     # Students will implement:
#     # 1. Load existing emails from data/emails.json
#     # 2. Add new email with topic and auto-increment ID
#     # 3. Save back to file
#     # 4. Return success response
#     pass

# TODO: Students will implement this endpoint  
# @router.get("/emails")
# async def get_all_emails():
#     """Get all stored emails from the database"""
#     # Students will implement:
#     # 1. Load emails from data/emails.json
#     # 2. Return list of all emails
#     pass

# LAB ASSIGNMENT: Add a new endpoint to list all available features
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system. Use the FeatureGeneratorFactory to get this information.