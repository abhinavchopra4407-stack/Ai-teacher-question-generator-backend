import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "TeachGenie AI"
    DEBUG: bool = True
    SECRET_KEY: str = os.getenv("SECRET_KEY", "teachgenie-super-secret-jwt-key-2026-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./teachgenie.db")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    MAX_UPLOAD_SIZE_MB: int = 15
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
