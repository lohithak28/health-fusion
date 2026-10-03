from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), unique=True, index=True, nullable=False)
    age = Column(Float, nullable=False)
    gender = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    predictions = relationship(
        "Prediction",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(Prediction.created_at)",
    )
    uploaded_files = relationship(
        "UploadedFile",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(UploadedFile.uploaded_at)",
    )

    def __repr__(self):
        return f"<Patient(id={self.id}, patient_id='{self.patient_id}', age={self.age}, gender='{self.gender}')>"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)

    cxr_prediction = Column(String(32), nullable=True)
    cxr_probability = Column(Float, nullable=True)

    multimodal_prediction = Column(String(32), nullable=False)
    multimodal_probability = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="predictions")
    uploaded_files = relationship("UploadedFile", back_populates="prediction")

    def __repr__(self):
        return (
            f"<Prediction(id={self.id}, patient_id={self.patient_id}, "
            f"pred='{self.multimodal_prediction}', prob={self.multimodal_probability:.4f})>"
        )


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True, index=True)

    file_type = Column(String(32), nullable=False)  # e.g., 'CXR', 'ECG', 'FEATURE'
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    mime_type = Column(String(64), nullable=True)
    file_size = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="uploaded_files")
    prediction = relationship("Prediction", back_populates="uploaded_files")

    def __repr__(self):
        return f"<UploadedFile(id={self.id}, type='{self.file_type}', path='{self.stored_path}')>"
