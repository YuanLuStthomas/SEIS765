# Email Topic Classification Lab - Factory Pattern

This lab demonstrates the Factory Pattern in machine learning feature generation and email topic classification using cosine similarity.

## Overview

The system classifies emails into topics (work, personal, promotion, newsletter, support) using:
- **Factory Pattern** for feature generation
- **Embedding-based similarity** using cosine distance
- **RESTful API** for classification and data management

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Run on all interfaces (required for EC2 access)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Access the API
- **Local**: http://localhost:8000
- **EC2**: http://YOUR_EC2_PUBLIC_IP:8000
- **API Documentation (Swagger)**: http://YOUR_EC2_PUBLIC_IP:8000/docs
- **Alternative Docs (ReDoc)**: http://YOUR_EC2_PUBLIC_IP:8000/redoc

## Getting Started - Explore the System

### 1. View Available Topics
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/topics
```

### 2. Classify an Email
```bash
curl -X POST "http://YOUR_EC2_PUBLIC_IP:8000/emails/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Meeting tomorrow at 2pm",
    "body": "Let'\''s discuss the quarterly budget and project deadlines"
  }'
```

### 3. Get Pipeline Information
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/pipeline/info
```

### 4. Interactive API Documentation
Visit `http://YOUR_EC2_PUBLIC_IP:8000/docs` in your browser for:
- Interactive API testing
- Request/response examples
- Schema definitions

## Understanding the Architecture

### Factory Pattern Implementation
- **Location**: `app/features/factory.py`
- **Generators**: `app/features/generators.py` 
- **Pattern**: See `GENERATORS` constant for available feature generators

### Feature Generators
1. **SpamFeatureGenerator** - Detects spam keywords
2. **AverageWordLengthFeatureGenerator** - Calculates average word length
3. **EmailEmbeddingsFeatureGenerator** - Creates embeddings from email length
4. **RawEmailFeatureGenerator** - Extracts raw email text

### Classification Model
- **Location**: `app/models/similarity_model.py`
- **Method**: Cosine similarity between email embeddings and topic description embeddings
- **Topics**: Stored in `data/topic_keywords.json`

## Key Files to Examine

1. **`app/features/factory.py`** - Factory pattern implementation
2. **`app/features/generators.py`** - Feature generator classes
3. **`app/models/similarity_model.py`** - Classification logic
4. **`app/api/routes.py`** - REST API endpoints
5. **`data/topic_keywords.json`** - Topic definitions and descriptions

## Your Assignment

Implement two new REST endpoints in `app/api/routes.py`:

### 1. Add Email to Database
```python
@router.post("/emails", response_model=EmailAddResponse)
async def add_email(request: EmailWithTopicRequest):
    # TODO: Implement this endpoint
    # 1. Load existing emails from data/emails.json
    # 2. Add new email with topic and auto-increment ID  
    # 3. Save back to file
    # 4. Return success response
```

### 2. Get All Emails
```python
@router.get("/emails")
async def get_all_emails():
    # TODO: Implement this endpoint
    # 1. Load emails from data/emails.json
    # 2. Return list of all emails
```

### Expected Email Storage Format
```json
[
  {
    "id": 1,
    "subject": "Meeting tomorrow",
    "body": "Let's discuss the project", 
    "topic": "work"
  }
]
```

## Testing Your Implementation

### Test Adding an Email
```bash
curl -X POST "http://YOUR_EC2_PUBLIC_IP:8000/emails" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Family dinner this weekend",
    "body": "Join us for dinner on Sunday at 6pm",
    "topic": "personal"
  }'
```

### Test Getting All Emails  
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/emails
```

## Learning Objectives

- Understand the **Factory Pattern** for extensible feature generation
- Learn **embedding-based similarity** for classification
- Practice **REST API design** following proper conventions
- Implement **file-based data persistence** as a database substitute
- Experience **machine learning pipeline** architecture

## Troubleshooting

- **Can't access from browser**: Make sure you're running with `--host 0.0.0.0`
- **Port issues**: Check that port 8000 is open in your EC2 security group
- **JSON errors**: Use the Swagger docs at `/docs` for proper request format
- **File permissions**: Ensure the `data/` directory is writable

Good luck with your implementation!
