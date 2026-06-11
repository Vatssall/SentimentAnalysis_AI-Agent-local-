# src/service/models.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .db import Base

class TherapyPlan(Base):
    __tablename__ = "therapy_plans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    title = Column(String(200), default="Personal Plan")
    created_at = Column(DateTime, default=datetime.utcnow)

    tasks = relationship("TherapyTask", back_populates="plan", cascade="all,delete-orphan")
    checkins = relationship("CheckIn", back_populates="plan", cascade="all,delete-orphan")

class TherapyTask(Base):
    __tablename__ = "therapy_tasks"
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("therapy_plans.id"))
    code = Column(String(50))
    title = Column(String(200))
    description = Column(Text)
    minutes = Column(Integer, default=10)
    done = Column(Boolean, default=False)

    plan = relationship("TherapyPlan", back_populates="tasks")

class CheckIn(Base):
    __tablename__ = "checkins"
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("therapy_plans.id"))
    mood = Column(String(30))
    emotion = Column(String(30))
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    plan = relationship("TherapyPlan", back_populates="checkins")