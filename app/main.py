from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ML Server with Feature Generation Factory",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)