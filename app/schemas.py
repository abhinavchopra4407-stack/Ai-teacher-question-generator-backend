from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth Schemas ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    custom_gemini_api_key: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None

class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    is_verified: bool
    has_custom_key: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# --- Chapter / Document Input Schemas ---
class TextExtractResponse(BaseModel):
    extracted_text: str
    word_count: int
    file_name: Optional[str] = None
    file_type: Optional[str] = None

# --- Question Schemas ---
class SingleQuestion(BaseModel):
    id: str
    question_number: int
    question_text: str
    question_type: str  # "Very Short Answer", "Short Answer", "Long Answer"
    difficulty: str     # "Easy", "Medium", "Hard"
    marks: int
    related_topic: str
    answer: Optional[str] = None
    marking_points: Optional[List[str]] = []
    expected_length: Optional[str] = None

class GenerateQuestionsRequest(BaseModel):
    document_id: Optional[str] = None
    chapter_title: str
    subject: str
    grade: str
    board: Optional[str] = "General / CBSE"
    language: str = "English"  # "English" or "Hindi"
    difficulty: str = "Medium" # "Easy", "Medium", "Hard"
    marks_distribution: Optional[Dict[str, int]] = {
        "very_short": 2,
        "short": 4,
        "long": 8
    }
    special_instructions: Optional[str] = None
    raw_content: Optional[str] = None

class QuestionPaperResponse(BaseModel):
    chapter_title: str
    subject: str
    grade: str
    board: str
    language: str
    difficulty: str
    total_marks: int
    very_short_questions: List[SingleQuestion]
    short_questions: List[SingleQuestion]
    long_questions: List[SingleQuestion]

class RegenerateSingleRequest(BaseModel):
    chapter_title: str
    subject: str
    grade: str
    language: str
    difficulty: str
    question_type: str
    existing_question: str
    topic: Optional[str] = ""
    chapter_text: str
    special_instructions: Optional[str] = None

class GenerateAnswerKeyRequest(BaseModel):
    chapter_title: str
    chapter_text: str
    questions: List[SingleQuestion]
    language: str = "English"

# --- Saved Paper Schemas ---
class SavePaperRequest(BaseModel):
    id: Optional[str] = None
    document_id: Optional[str] = None
    title: str
    subject: str
    grade: str
    board: Optional[str] = "General"
    language: str = "English"
    difficulty: str = "Medium"
    total_marks: int
    school_name: Optional[str] = "TeachGenie Model School"
    teacher_name: Optional[str] = ""
    instructions: Optional[str] = "Attempt all questions. Read instructions carefully."
    questions: List[SingleQuestion]
    answer_key: Optional[List[SingleQuestion]] = None

class QuestionPaperOut(BaseModel):
    id: str
    user_id: str
    document_id: Optional[str] = None
    title: str
    subject: str
    grade: str
    board: Optional[str]
    language: str
    difficulty: str
    total_marks: int
    school_name: Optional[str]
    teacher_name: Optional[str]
    instructions: Optional[str]
    questions: List[SingleQuestion]
    answer_key: Optional[List[SingleQuestion]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_papers: int
    total_chapters: int
    recent_papers: List[QuestionPaperOut]
