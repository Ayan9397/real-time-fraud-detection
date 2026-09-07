import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "fraud_detection")


if not DB_PASSWORD:
    raise RuntimeError(
        "POSTGRES_PASSWORD is not configured. "
        "Add it to the project's .env file."
    )


# ---------------------------------------------------------
# Build database URL safely
# ---------------------------------------------------------
#
# URL.create() is intentionally used instead of manually
# constructing a connection string.
#
# This correctly handles special characters in passwords,
# such as:
#
# @ : / # % ? !
#
# ---------------------------------------------------------

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)


# ---------------------------------------------------------
# SQLAlchemy engine
# ---------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)


# ---------------------------------------------------------
# SQLAlchemy declarative base
# ---------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------
# Database session factory
# ---------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------
# Database dependency
# ---------------------------------------------------------

def get_db():
    """
    Provide a SQLAlchemy database session.

    The session is automatically closed after use.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------
# Database connection test
# ---------------------------------------------------------

def test_connection() -> None:
    """
    Test connectivity to PostgreSQL.
    """

    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        version = result.scalar()

        print("PostgreSQL connection successful.")
        print(f"Database: {DB_NAME}")
        print(f"Host: {DB_HOST}")
        print(f"Port: {DB_PORT}")
        print(f"Server: {version}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    test_connection()