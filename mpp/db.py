"""SQLAlchemy models, engine and session for the standardized dataset (SQLite)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from . import config
from .lipids import SEED_LIPIDS


class Base(DeclarativeBase):
    pass


class Lipid(Base):
    __tablename__ = "lipids"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(32))
    full_name: Mapped[str] = mapped_column(String(256), default="")
    notes: Mapped[str] = mapped_column(Text, default="")


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # Full CampaignConfig.model_dump() lives here.
    config: Mapped[dict] = mapped_column(JSON)

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan"
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"), index=True)
    label: Mapped[str] = mapped_column(String(128), default="")
    plate: Mapped[str] = mapped_column(String(32), default="")
    well: Mapped[str] = mapped_column(String(8), default="")
    # source: suggested | manual | demo ; status: suggested | completed
    source: Mapped[str] = mapped_column(String(16), default="manual")
    status: Mapped[str] = mapped_column(String(16), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")

    composition: Mapped[dict] = mapped_column(JSON, default=dict)  # {lipid: mole_fraction}
    process: Mapped[dict] = mapped_column(JSON, default=dict)      # {param: value}
    readouts: Mapped[dict] = mapped_column(JSON, default=dict)     # {readout: value}

    campaign: Mapped["Campaign"] = relationship(back_populates="experiments")
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    filename: Mapped[str] = mapped_column(String(256))
    path: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(128), default="")
    size: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship(back_populates="attachments")


_engine = create_engine(config.DB_URL, future=True)
SessionLocal = sessionmaker(bind=_engine, future=True, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator:
    """Session context manager that commits on success and rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(seed: bool = True) -> None:
    """Create tables and (optionally) seed the lipid library if empty."""
    Base.metadata.create_all(_engine)
    if not seed:
        return
    with get_session() as s:
        existing = s.query(func.count(Lipid.id)).scalar()
        if existing:
            return
        for l in SEED_LIPIDS:
            s.add(Lipid(name=l["name"], category=l["category"], full_name=l["full_name"], notes=l["notes"]))
