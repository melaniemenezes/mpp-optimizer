"""Shared Streamlit helpers used across pages."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from . import service


def ensure_init() -> None:
    if not st.session_state.get("_initialized"):
        service.init_app()
        st.session_state["_initialized"] = True


def campaign_selectbox() -> Optional[dict]:
    """Sidebar selector for the active campaign. Returns the campaign dict or None."""
    camps = service.list_campaigns()
    if not camps:
        st.sidebar.info("No campaigns yet — create one on the **Campaign Setup** page.")
        return None
    ids = [c["id"] for c in camps]
    labels = {c["id"]: f"{c['name']}  (#{c['id']})" for c in camps}
    current = st.session_state.get("campaign_id", ids[0])
    if current not in ids:
        current = ids[0]
    sel = st.sidebar.selectbox(
        "Active campaign", ids, index=ids.index(current), format_func=lambda i: labels[i]
    )
    st.session_state["campaign_id"] = sel
    return service.get_campaign(sel)


def require_campaign() -> dict:
    ensure_init()
    camp = campaign_selectbox()
    if camp is None:
        st.warning("Create a campaign first on the **Campaign Setup** page.")
        st.stop()
    return camp


def composition_summary(comp: dict, top: int = 4) -> str:
    items = sorted(comp.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k} {v:.2f}" for k, v in items)
