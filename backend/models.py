from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean
)

from datetime import datetime
from database import Base

class IncidentLog(Base):

    __tablename__ = "incident_logs"

    id = Column(Integer, primary_key=True)

    incident_id = Column(String, unique=True)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow
    )

    location = Column(String)

    object_involved = Column(String)

    threat_level = Column(String)

    protocol_number = Column(String)

    status = Column(String)

    requires_immediate_action = Column(Boolean)
