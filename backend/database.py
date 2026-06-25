# backend/database.py (नवीन फाईल)
import os
from datetime import datetime
from sqlalchemy import create_backend, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# स्थानिक चाचणीसाठी SQLite वापरू शकता, प्रोडक्शनसाठी PostgreSQL URL वापरा
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./incidents.db")

engine = create_backend(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class IncidentLog(Base):
    __tablename__ = "incident_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    location = Column(String)
    object_involved = Column(String)
    threat_level = Column(String)
    confidence_score = Column(String)
    protocol_number = Column(String, nullable=True)
    requires_immediate_action = Column(Boolean)

# डेटाबेस टेबल्स तयार करणे
Base.metadata.create_all(bind=engine)

# FastAPI साठी DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
