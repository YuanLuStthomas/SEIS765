from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ML Server"
    DEBUG: bool = True
    MODEL_PATH: Optional[str] = None
    FEATURE_STORE_PATH: str = "data/"


settings = Settings()