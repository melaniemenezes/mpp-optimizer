"""Shared Streamlit helpers used across pages."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from . import service

# --------------------------------------------------------------------------- theme
# A shadcn/ui-inspired design language (neutral "zinc" palette, Inter, bordered rounded
# cards, muted text) applied globally via CSS. Streamlit is React under the hood but not
# extensible with shadcn's own components, so we bring its *look* to native widgets.
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --mpp-bg: #ffffff;
  --mpp-fg: #09090b;
  --mpp-muted: #f4f4f5;
  --mpp-muted-fg: #71717a;
  --mpp-border: #e4e4e7;
  --mpp-primary: #18181b;
  --mpp-radius: 0.65rem;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.stApp { background: var(--mpp-bg); }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1120px; }

h1, h2, h3, h4 { color: var(--mpp-fg); font-weight: 600; letter-spacing: -0.02em; }
h1 { font-size: 1.85rem; }
[data-testid="stCaptionContainer"] p { color: var(--mpp-muted-fg) !important; }

/* bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--mpp-border) !important;
  border-radius: var(--mpp-radius) !important;
  background: #fff;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}

/* metrics -> stat cards */
[data-testid="stMetric"] {
  background: #fff;
  border: 1px solid var(--mpp-border);
  border-radius: var(--mpp-radius);
  padding: 0.9rem 1.1rem;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
[data-testid="stMetricLabel"] p { color: var(--mpp-muted-fg); font-weight: 500; }

/* buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 0.55rem;
  border: 1px solid var(--mpp-border);
  font-weight: 500;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
  transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover { border-color: #d4d4d8; background: var(--mpp-muted); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--mpp-primary); color: #fff; border-color: var(--mpp-primary);
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover { background: #27272a; border-color: #27272a; color: #fff; }

/* inputs */
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div, [data-baseweb="input"] {
  border-radius: 0.5rem !important;
}

/* sidebar */
[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid var(--mpp-border); }

/* tabs */
[data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--mpp-border); }
[data-baseweb="tab"] { font-weight: 500; }

/* dataframes & expanders -> cards */
[data-testid="stDataFrame"] { border: 1px solid var(--mpp-border); border-radius: var(--mpp-radius); overflow: hidden; }
[data-testid="stExpander"] {
  border: 1px solid var(--mpp-border) !important;
  border-radius: var(--mpp-radius) !important;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
hr { border-color: var(--mpp-border); }

/* helper classes */
.mpp-badge {
  display: inline-block; padding: 2px 10px; margin: 0 4px 4px 0; border-radius: 9999px;
  font-size: 0.75rem; font-weight: 500; border: 1px solid var(--mpp-border);
  background: var(--mpp-muted); color: #3f3f46;
}
.mpp-sub { color: var(--mpp-muted-fg); font-size: 1rem; margin-top: -0.35rem; }
</style>
"""


def apply_theme() -> None:
    """Inject the shadcn-inspired CSS. Call once per page, right after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: Optional[str] = None, icon: str = "") -> None:
    """A consistent, styled page title + subtitle."""
    label = f"{icon} {title}".strip()
    st.markdown(f"<h1 style='margin-bottom:0.25rem'>{label}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='mpp-sub'>{subtitle}</p>", unsafe_allow_html=True)
    st.write("")


def badges(*labels: str) -> None:
    """Render a row of pill badges."""
    html = "".join(f"<span class='mpp-badge'>{l}</span>" for l in labels if l)
    st.markdown(html, unsafe_allow_html=True)


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
