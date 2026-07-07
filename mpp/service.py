"""High-level operations used by the Streamlit pages and scripts.

Everything returns plain dicts / primitives (never live ORM objects) so results stay
valid across Streamlit reruns and detached sessions.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .db import Attachment, Campaign, Experiment, Lipid, get_session, init_db
from .schema import CampaignConfig, ExperimentRecord


def init_app() -> None:
    init_db(seed=True)


# ------------------------------------------------------------------- lipids
def list_lipids() -> List[dict]:
    with get_session() as s:
        rows = s.query(Lipid).order_by(Lipid.category, Lipid.name).all()
        return [
            {"name": r.name, "category": r.category, "full_name": r.full_name, "notes": r.notes}
            for r in rows
        ]


# ---------------------------------------------------------------- campaigns
def create_campaign(config: CampaignConfig) -> int:
    with get_session() as s:
        c = Campaign(name=config.name, description=config.description, config=config.model_dump())
        s.add(c)
        s.flush()
        return c.id


def list_campaigns() -> List[dict]:
    with get_session() as s:
        rows = s.query(Campaign).order_by(Campaign.created_at.desc()).all()
        out = []
        for r in rows:
            n = s.query(Experiment).filter_by(campaign_id=r.id).count()
            out.append({"id": r.id, "name": r.name, "created_at": r.created_at, "n_experiments": n})
        return out


def get_campaign(campaign_id: int) -> Optional[dict]:
    with get_session() as s:
        r = s.get(Campaign, campaign_id)
        if r is None:
            return None
        return {"id": r.id, "name": r.name, "description": r.description,
                "created_at": r.created_at, "config": CampaignConfig(**r.config)}


def delete_campaign(campaign_id: int) -> None:
    with get_session() as s:
        r = s.get(Campaign, campaign_id)
        if r is not None:
            s.delete(r)


# -------------------------------------------------------------- experiments
def _exp_to_dict(e: Experiment) -> dict:
    return {
        "id": e.id,
        "campaign_id": e.campaign_id,
        "label": e.label,
        "plate": e.plate,
        "well": e.well,
        "source": e.source,
        "status": e.status,
        "created_at": e.created_at,
        "notes": e.notes,
        "composition": dict(e.composition or {}),
        "process": dict(e.process or {}),
        "readouts": dict(e.readouts or {}),
        "attachments": [
            {"id": a.id, "filename": a.filename, "path": a.path,
             "content_type": a.content_type, "size": a.size}
            for a in e.attachments
        ],
    }


def add_experiment(campaign_id: int, record: ExperimentRecord, source: str = "manual",
                   status: str = "completed") -> int:
    with get_session() as s:
        e = Experiment(
            campaign_id=campaign_id, label=record.label, plate=record.plate, well=record.well,
            source=source, status=status, notes=record.notes,
            composition=record.composition, process=record.process, readouts=record.readouts,
        )
        s.add(e)
        s.flush()
        return e.id


def update_experiment(exp_id: int, *, composition=None, process=None, readouts=None,
                      label=None, plate=None, well=None, notes=None, status=None) -> None:
    with get_session() as s:
        e = s.get(Experiment, exp_id)
        if e is None:
            return
        if composition is not None:
            e.composition = composition
        if process is not None:
            e.process = process
        if readouts is not None:
            e.readouts = readouts
        if label is not None:
            e.label = label
        if plate is not None:
            e.plate = plate
        if well is not None:
            e.well = well
        if notes is not None:
            e.notes = notes
        if status is not None:
            e.status = status


def list_experiments(campaign_id: int, status: Optional[str] = None) -> List[dict]:
    with get_session() as s:
        q = s.query(Experiment).filter_by(campaign_id=campaign_id)
        if status:
            q = q.filter_by(status=status)
        return [_exp_to_dict(e) for e in q.order_by(Experiment.created_at).all()]


def get_experiment(exp_id: int) -> Optional[dict]:
    with get_session() as s:
        e = s.get(Experiment, exp_id)
        return _exp_to_dict(e) if e else None


def delete_experiment(exp_id: int) -> None:
    with get_session() as s:
        e = s.get(Experiment, exp_id)
        if e is not None:
            s.delete(e)


def add_attachment(exp_id: int, meta: dict) -> None:
    with get_session() as s:
        s.add(Attachment(
            experiment_id=exp_id, filename=meta["filename"], path=meta["path"],
            content_type=meta.get("content_type", ""), size=meta.get("size", 0),
        ))


# ----------------------------------------------------- optimizer / dataframe
def records_for_optimizer(campaign_id: int) -> List[dict]:
    """All experiments as {composition, process, readouts} dicts (optimizer filters completeness)."""
    return [
        {"composition": e["composition"], "process": e["process"], "readouts": e["readouts"], "id": e["id"]}
        for e in list_experiments(campaign_id)
    ]


def experiments_dataframe(campaign_id: int) -> pd.DataFrame:
    exps = list_experiments(campaign_id)
    rows = []
    for e in exps:
        row = {
            "id": e["id"], "label": e["label"], "plate": e["plate"], "well": e["well"],
            "source": e["source"], "status": e["status"],
            "created_at": e["created_at"], "n_files": len(e["attachments"]),
        }
        for k, v in e["composition"].items():
            row[f"x:{k}"] = v
        for k, v in e["process"].items():
            row[f"p:{k}"] = v
        for k, v in e["readouts"].items():
            row[f"y:{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
