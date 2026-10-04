import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

db_url = settings.DATABASE_URL or "sqlite:///./teachgenie.db"

# Convert legacy 'postgres://' to 'postgresql://' for SQLAlchemy 1.4+ / 2.0 compatibility
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    if ":///" in db_url:
        sqlite_path = db_url.split(":///", 1)[1]
        if sqlite_path and not sqlite_path.startswith(":memory:"):
            dir_name = os.path.dirname(os.path.abspath(sqlite_path))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
