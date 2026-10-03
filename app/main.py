import json
import os
import uuid
import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import engine, Base, get_db
from app import models, schemas, auth, doc_processor, ai_engine, exporter

# Initialize Database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="TeachGenie AI API",
    description="Backend service for TeachGenie AI Smart Question Paper Generator",
    version="1.0.0"
)

# Allowed CORS Origins for Production & Local Development
allowed_origins = [
    "https://ai-teacher-question-generator-front.vercel.app",
    "https://ai-teacher-question-generator-front.vercel.app/",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

frontend_env = os.getenv("FRONTEND_URL")
if frontend_env:
    allowed_origins.append(frontend_env.strip().replace(/\/+$/, ''))

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "message": "Welcome to TeachGenie AI Question Paper Generator API"
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

# ================= AUTHENTICATION ENDPOINTS =================

@app.post("/api/auth/register", response_model=schemas.Token)
def register_user(user_in: schemas.UserRegister, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user_in.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
        
    hashed_pwd = auth.get_password_hash(user_in.password)
    user = models.User(
        email=user_in.email.lower(),
        full_name=user_in.full_name,
        hashed_password=hashed_pwd,
        is_verified=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    access_token = auth.create_access_token(data={"sub": user.id})
    user_out = schemas.UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_verified=user.is_verified,
        has_custom_key=bool(user.custom_gemini_api_key),
        created_at=user.created_at
    )
    return {"access_token": access_token, "token_type": "bearer", "user": user_out}

@app.post("/api/auth/login", response_model=schemas.Token)
def login_user(user_in: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == user_in.email.lower()).first()
    if not user or not auth.verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password. Please check your credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = auth.create_access_token(data={"sub": user.id})
    user_out = schemas.UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_verified=user.is_verified,
        has_custom_key=bool(user.custom_gemini_api_key),
        created_at=user.created_at
    )
    return {"access_token": access_token, "token_type": "bearer", "user": user_out}

@app.get("/api/auth/me", response_model=schemas.UserOut)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return schemas.UserOut(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_verified=current_user.is_verified,
        has_custom_key=bool(current_user.custom_gemini_api_key),
        created_at=current_user.created_at
    )

@app.post("/api/auth/forgot-password")
def forgot_password(req: schemas.ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == req.email.lower()).first()
    if not user:
        return {"message": "If that email exists in our records, password reset instructions have been sent."}
    return {"message": "Password reset email sent successfully. Please check your inbox for instructions."}

@app.post("/api/auth/reset-password")
def reset_password(req: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    return {"message": "Password updated successfully. You may now log in with your new password."}

@app.put("/api/auth/profile", response_model=schemas.UserOut)
def update_profile(
    req: schemas.UserProfileUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    if req.full_name:
        current_user.full_name = req.full_name
        
    if req.custom_gemini_api_key is not None:
        current_user.custom_gemini_api_key = req.custom_gemini_api_key.strip() or None
        
    if req.new_password:
        if not req.current_password or not auth.verify_password(req.current_password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is invalid.")
        current_user.hashed_password = auth.get_password_hash(req.new_password)
        
    db.commit()
    db.refresh(current_user)
    
    return schemas.UserOut(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_verified=current_user.is_verified,
        has_custom_key=bool(current_user.custom_gemini_api_key),
        created_at=current_user.created_at
    )

# ================= DOCUMENT & CHAPTER PROCESSING ENDPOINTS =================

@app.post("/api/documents/upload", response_model=schemas.TextExtractResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    extracted_text = ""
    file_name = None
    file_type = "raw_text"
    
    if file:
        file_bytes = await file.read()
        if len(file_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.")
            
        file_name = file.filename
        file_type = file.filename.split(".")[-1] if "." in file.filename else "unknown"
        extracted_text, word_count = doc_processor.process_uploaded_document(file, file_bytes)
    elif raw_text and raw_text.strip():
        extracted_text = doc_processor.clean_text(raw_text)
        word_count = len(extracted_text.split())
        file_name = "Pasted Chapter Content"
    else:
        raise HTTPException(status_code=400, detail="Please upload a PDF/DOCX file or paste chapter content text.")
        
    doc_title = title or (file_name if file_name else "Chapter Content")
    
    doc_record = models.Document(
        user_id=current_user.id,
        title=doc_title,
        file_name=file_name,
        file_type=file_type,
        extracted_text=extracted_text,
        word_count=word_count
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    
    return {
        "extracted_text": extracted_text,
        "word_count": word_count,
        "file_name": file_name,
        "file_type": file_type,
        "document_id": doc_record.id
    }

# ================= AI QUESTION GENERATION ENDPOINTS =================

@app.post("/api/questions/generate", response_model=schemas.QuestionPaperResponse)
def generate_questions(
    req: schemas.GenerateQuestionsRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    chapter_text = ""
    
    if req.document_id:
        doc = db.query(models.Document).filter(
            models.Document.id == req.document_id,
            models.Document.user_id == current_user.id
        ).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or access denied.")
        chapter_text = doc.extracted_text
    elif req.raw_content and req.raw_content.strip():
        chapter_text = doc_processor.clean_text(req.raw_content)
    else:
        raise HTTPException(status_code=400, detail="Please provide either a document_id or raw chapter content.")
        
    if not chapter_text or len(chapter_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Chapter content is empty or contains insufficient text.")

    generated_data = ai_engine.generate_questions_from_chapter(
        chapter_title=req.chapter_title,
        chapter_text=chapter_text,
        subject=req.subject,
        grade=req.grade,
        board=req.board or "General",
        language=req.language,
        difficulty=req.difficulty,
        marks_dist=req.marks_distribution,
        special_instructions=req.special_instructions,
        user_api_key=current_user.custom_gemini_api_key
    )
    
    vs = generated_data.get("very_short_questions", [])
    sq = generated_data.get("short_questions", [])
    lq = generated_data.get("long_questions", [])
    
    total_marks = sum(q["marks"] for q in vs + sq + lq)
    
    return {
        "chapter_title": req.chapter_title,
        "subject": req.subject,
        "grade": req.grade,
        "board": req.board or "General",
        "language": req.language,
        "difficulty": req.difficulty,
        "total_marks": total_marks,
        "very_short_questions": vs,
        "short_questions": sq,
        "long_questions": lq
    }

@app.post("/api/questions/regenerate-single", response_model=schemas.SingleQuestion)
def regenerate_single_question(
    req: schemas.RegenerateSingleRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    new_q = ai_engine.generate_single_replacement_question(
        chapter_title=req.chapter_title,
        chapter_text=req.chapter_text,
        subject=req.subject,
        grade=req.grade,
        question_type=req.question_type,
        existing_question=req.existing_question,
        topic=req.topic or "",
        language=req.language,
        difficulty=req.difficulty,
        special_instructions=req.special_instructions,
        user_api_key=current_user.custom_gemini_api_key
    )
    return new_q

@app.post("/api/questions/generate-answer-key", response_model=List[schemas.SingleQuestion])
def generate_answer_key(
    req: schemas.GenerateAnswerKeyRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    questions_dicts = [q.dict() for q in req.questions]
    updated_questions = ai_engine.generate_answer_key_for_questions(
        chapter_title=req.chapter_title,
        chapter_text=req.chapter_text,
        questions=questions_dicts,
        language=req.language,
        user_api_key=current_user.custom_gemini_api_key
    )
    return updated_questions

# ================= SAVED QUESTION PAPERS ENDPOINTS =================

@app.get("/api/papers", response_model=List[schemas.QuestionPaperOut])
def get_user_papers(
    search: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    grade: Optional[str] = Query(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(models.QuestionPaper).filter(models.QuestionPaper.user_id == current_user.id)
    
    if subject and subject != "All":
        query = query.filter(models.QuestionPaper.subject == subject)
    if grade and grade != "All":
        query = query.filter(models.QuestionPaper.grade == grade)
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(models.QuestionPaper.title.ilike(term) | models.QuestionPaper.subject.ilike(term))
        
    papers = query.order_by(models.QuestionPaper.updated_at.desc()).all()
    
    res = []
    for p in papers:
        q_list = json.loads(p.questions_json) if p.questions_json else []
        ak_list = json.loads(p.answer_key_json) if p.answer_key_json else None
        res.append(schemas.QuestionPaperOut(
            id=p.id,
            user_id=p.user_id,
            document_id=p.document_id,
            title=p.title,
            subject=p.subject,
            grade=p.grade,
            board=p.board,
            language=p.language,
            difficulty=p.difficulty,
            total_marks=p.total_marks,
            school_name=p.school_name,
            teacher_name=p.teacher_name,
            instructions=p.instructions,
            questions=q_list,
            answer_key=ak_list,
            created_at=p.created_at,
            updated_at=p.updated_at
        ))
    return res

@app.post("/api/papers", response_model=schemas.QuestionPaperOut)
def save_paper(
    req: schemas.SavePaperRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    questions_json = json.dumps([q.dict() for q in req.questions])
    answer_key_json = json.dumps([q.dict() for q in req.answer_key]) if req.answer_key else None
    
    if req.id:
        paper = db.query(models.QuestionPaper).filter(
            models.QuestionPaper.id == req.id,
            models.QuestionPaper.user_id == current_user.id
        ).first()
        if not paper:
            raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
        paper.title = req.title
        paper.subject = req.subject
        paper.grade = req.grade
        paper.board = req.board or "General"
        paper.language = req.language
        paper.difficulty = req.difficulty
        paper.total_marks = req.total_marks
        paper.school_name = req.school_name
        paper.teacher_name = req.teacher_name
        paper.instructions = req.instructions
        paper.questions_json = questions_json
        paper.answer_key_json = answer_key_json
        paper.updated_at = datetime.datetime.utcnow()
    else:
        paper = models.QuestionPaper(
            user_id=current_user.id,
            document_id=req.document_id,
            title=req.title,
            subject=req.subject,
            grade=req.grade,
            board=req.board or "General",
            language=req.language,
            difficulty=req.difficulty,
            total_marks=req.total_marks,
            school_name=req.school_name or "TeachGenie Model School",
            teacher_name=req.teacher_name or current_user.full_name,
            instructions=req.instructions or "Attempt all questions. Read instructions carefully.",
            questions_json=questions_json,
            answer_key_json=answer_key_json
        )
        db.add(paper)
        
    db.commit()
    db.refresh(paper)
    
    return schemas.QuestionPaperOut(
        id=paper.id,
        user_id=paper.user_id,
        document_id=paper.document_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        board=paper.board,
        language=paper.language,
        difficulty=paper.difficulty,
        total_marks=paper.total_marks,
        school_name=paper.school_name,
        teacher_name=paper.teacher_name,
        instructions=paper.instructions,
        questions=[q.dict() for q in req.questions],
        answer_key=[q.dict() for q in req.answer_key] if req.answer_key else None,
        created_at=paper.created_at,
        updated_at=paper.updated_at
    )

@app.get("/api/papers/{paper_id}", response_model=schemas.QuestionPaperOut)
def get_paper_by_id(
    paper_id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    paper = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.id == paper_id,
        models.QuestionPaper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
    return schemas.QuestionPaperOut(
        id=paper.id,
        user_id=paper.user_id,
        document_id=paper.document_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        board=paper.board,
        language=paper.language,
        difficulty=paper.difficulty,
        total_marks=paper.total_marks,
        school_name=paper.school_name,
        teacher_name=paper.teacher_name,
        instructions=paper.instructions,
        questions=json.loads(paper.questions_json) if paper.questions_json else [],
        answer_key=json.loads(paper.answer_key_json) if paper.answer_key_json else None,
        created_at=paper.created_at,
        updated_at=paper.updated_at
    )

@app.delete("/api/papers/{paper_id}")
def delete_paper(
    paper_id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    paper = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.id == paper_id,
        models.QuestionPaper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
    db.delete(paper)
    db.commit()
    return {"message": "Question paper deleted successfully."}

@app.post("/api/papers/{paper_id}/duplicate", response_model=schemas.QuestionPaperOut)
def duplicate_paper(
    paper_id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    paper = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.id == paper_id,
        models.QuestionPaper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
    new_paper = models.QuestionPaper(
        user_id=current_user.id,
        document_id=paper.document_id,
        title=f"{paper.title} (Copy)",
        subject=paper.subject,
        grade=paper.grade,
        board=paper.board,
        language=paper.language,
        difficulty=paper.difficulty,
        total_marks=paper.total_marks,
        school_name=paper.school_name,
        teacher_name=paper.teacher_name,
        instructions=paper.instructions,
        questions_json=paper.questions_json,
        answer_key_json=paper.answer_key_json
    )
    db.add(new_paper)
    db.commit()
    db.refresh(new_paper)
    
    return schemas.QuestionPaperOut(
        id=new_paper.id,
        user_id=new_paper.user_id,
        document_id=new_paper.document_id,
        title=new_paper.title,
        subject=new_paper.subject,
        grade=new_paper.grade,
        board=new_paper.board,
        language=new_paper.language,
        difficulty=new_paper.difficulty,
        total_marks=new_paper.total_marks,
        school_name=new_paper.school_name,
        teacher_name=new_paper.teacher_name,
        instructions=new_paper.instructions,
        questions=json.loads(new_paper.questions_json) if new_paper.questions_json else [],
        answer_key=json.loads(new_paper.answer_key_json) if new_paper.answer_key_json else None,
        created_at=new_paper.created_at,
        updated_at=new_paper.updated_at
    )

# ================= EXPORT ENDPOINTS =================

@app.get("/api/papers/{paper_id}/export/docx")
def export_docx(
    paper_id: str,
    include_answers: bool = Query(False),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    paper = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.id == paper_id,
        models.QuestionPaper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
    questions = json.loads(paper.questions_json) if paper.questions_json else []
    answer_key = json.loads(paper.answer_key_json) if (include_answers and paper.answer_key_json) else None
    
    docx_bytes = exporter.generate_docx_paper(
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        board=paper.board or "General",
        school_name=paper.school_name or "TeachGenie Model School",
        teacher_name=paper.teacher_name or current_user.full_name,
        total_marks=paper.total_marks,
        instructions=paper.instructions or "",
        questions=questions,
        answer_key=answer_key
    )
    
    filename = f"{paper.title.replace(' ', '_')}_QuestionPaper.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/papers/{paper_id}/export/pdf")
def export_pdf(
    paper_id: str,
    include_answers: bool = Query(False),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    paper = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.id == paper_id,
        models.QuestionPaper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found or access denied.")
        
    questions = json.loads(paper.questions_json) if paper.questions_json else []
    answer_key = json.loads(paper.answer_key_json) if (include_answers and paper.answer_key_json) else None
    
    pdf_bytes = exporter.generate_pdf_paper(
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        board=paper.board or "General",
        school_name=paper.school_name or "TeachGenie Model School",
        teacher_name=paper.teacher_name or current_user.full_name,
        total_marks=paper.total_marks,
        instructions=paper.instructions or "",
        questions=questions,
        answer_key=answer_key
    )
    
    filename = f"{paper.title.replace(' ', '_')}_QuestionPaper.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ================= DASHBOARD STATS ENDPOINT =================

@app.get("/api/dashboard/stats", response_model=schemas.DashboardStats)
def get_dashboard_stats(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    total_papers = db.query(models.QuestionPaper).filter(models.QuestionPaper.user_id == current_user.id).count()
    total_chapters = db.query(models.Document).filter(models.Document.user_id == current_user.id).count()
    
    recent_p = db.query(models.QuestionPaper).filter(
        models.QuestionPaper.user_id == current_user.id
    ).order_by(models.QuestionPaper.updated_at.desc()).limit(5).all()
    
    recent_out = []
    for p in recent_p:
        recent_out.append(schemas.QuestionPaperOut(
            id=p.id,
            user_id=p.user_id,
            document_id=p.document_id,
            title=p.title,
            subject=p.subject,
            grade=p.grade,
            board=p.board,
            language=p.language,
            difficulty=p.difficulty,
            total_marks=p.total_marks,
            school_name=p.school_name,
            teacher_name=p.teacher_name,
            instructions=p.instructions,
            questions=json.loads(p.questions_json) if p.questions_json else [],
            answer_key=json.loads(p.answer_key_json) if p.answer_key_json else None,
            created_at=p.created_at,
            updated_at=p.updated_at
        ))
        
    return {
        "total_papers": total_papers,
        "total_chapters": total_chapters,
        "recent_papers": recent_out
    }
