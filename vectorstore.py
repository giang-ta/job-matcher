# vectorstore.py

from typing import List
from sqlalchemy import create_engine, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.dialects.postgresql import ARRAY


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str]
    content: Mapped[str]
    documentId: Mapped[str]
    url: Mapped[str]
    embedding: Mapped[List[float]] = mapped_column(ARRAY(Float))


def create_documents(documents: list):
    engine = create_engine(
        "postgresql://postgres:admin@localhost:5433/app",
        echo=True,
    )
    with Session(engine) as session:
        session.add_all(documents)
        session.commit()
