import os
from pathlib import Path
from typing import Generator, Optional
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, Session
from backend.database.models import Base

# Automatically read local .env if present without requiring external packages
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    try:
        with open(_ENV_PATH, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass

DEFAULT_POSTGRES_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/healthfusion"


def get_database_url() -> str:
    """Retrieves the database connection URL from environment, or defaults to local PostgreSQL."""
    return os.environ.get("DATABASE_URL", DEFAULT_POSTGRES_URL)


def create_db_engine(database_url: Optional[str] = None):
    """Creates a SQLAlchemy engine configured with pre-ping connection health check."""
    url = database_url or get_database_url()
    
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


# Module-level engine: initialized immediately with DATABASE_URL
try:
    _engine = create_db_engine()
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
except Exception:
    _engine = None
    _SessionLocal = None


def get_engine(database_url: Optional[str] = None):
    """Creates or returns the active SQLAlchemy engine."""
    global _engine
    if database_url:
        return create_db_engine(database_url)
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def init_db(database_url: Optional[str] = None, create_tables: bool = True) -> sessionmaker:
    """
    Initializes the engine and session factory.
    Creates tables if create_tables=True and database is accessible.
    """
    global _engine, _SessionLocal
    _engine = get_engine(database_url)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    if create_tables:
        Base.metadata.create_all(bind=_engine)

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session per request.
    Raises HTTP 503 if database connection cannot be established.
    """
    global _SessionLocal
    try:
        if _SessionLocal is None:
            init_db()
        db: Session = _SessionLocal()
    except (OperationalError, Exception) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {str(e)}. Please check DATABASE_URL configuration.",
        )

    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_optional() -> Generator[Optional[Session], None, None]:
    """
    Optional database dependency for endpoints that work without persistence
    when DATABASE_URL is not yet configured or reachable.
    """
    global _SessionLocal
    try:
        if _SessionLocal is None:
            init_db()
        db: Session = _SessionLocal()
    except Exception:
        yield None
        return

    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
