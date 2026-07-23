"""Page 1 — define / manage optimization campaigns."""
import pandas as pd
import streamlit as st

from mpp import service
from mpp.benchmark import make_demo_campaign, make_diffusion_study_campaign
from mpp.config import STANDARD_READOUTS
from mpp.schema import (
    CampaignConfig,
    ComponentSpec,
    ConstraintSpec,
    ObjectiveSpec,
    ProcessParamSpec,
)
from mpp.ui import apply_theme, ensure_init, page_header

st.set_page_config(page_title="Campaign Setup · MPP", page_icon="🧫", layout="wide")
apply_theme()
ensure_init()
page_header("Campaign Setup", "Define the lipids, input features, objectives and constraints for a campaign.", icon="🧪")

READOUT_NAMES = [r[0] for r in STANDARD_READOUTS]
READOUT_DEFAULT_DIR = {r[0]: r[2] for r in STANDARD_READOUTS}
lipids = service.list_lipids()
lipid_names = [l["name"] for l in lipids]
lipid_cat = {l["name"]: l["category"] for l in lipids}

# ----------------------------------------------------------------- quick start
with st.expander("⚡ Quick start: load a ready-made campaign", expanded=False):
    qc1, qc2 = st.columns(2)
    with qc1:
        st.markdown("**Diffusion study** — the 5-lipid spec")
        st.caption("DDAB / DSPG / HSPC / Cholesterol / mPEG molar ratios + size + zeta as inputs; "
                   "MPT diffusion metrics (D, D1, Dalpha, alpha, net-to-path) as outputs.")
        if st.button("Create diffusion-study campaign"):
            cid = service.create_campaign(make_diffusion_study_campaign())
            st.session_state["campaign_id"] = cid
            st.success(f"Created diffusion-study campaign #{cid}. Enter data on **Upload Results**, "
                       f"then characterise each output on **Model & Insights**.")
    with qc2:
        st.markdown("**Generic demo** — optimizer example")
        st.caption("DSPC / Cholesterol / PEG / DOTAP campaign for the formulation-optimization demo.")
        if st.button("Create demo campaign"):
            cid = service.create_campaign(make_demo_campaign())
            st.session_state["campaign_id"] = cid
            st.success(f"Created demo campaign #{cid}. Open **Suggest Experiments** next.")

st.divider()
st.subheader("Create a new campaign")

name = st.text_input("Campaign name", value="My MPP campaign")
description = st.text_area("Description", value="", height=70)

# ---- components ----
st.markdown("##### 1. Lipid components")
chosen = st.multiselect(
    "Lipids in this campaign", lipid_names,
    default=[n for n in ["DSPC", "Cholesterol", "DSPE-PEG2000"] if n in lipid_names],
    help="Pick the lipids you want to vary. One must be the structural / filler lipid (it absorbs "
         "the remaining mole fraction so the recipe always sums to 1).",
)
comp_df = pd.DataFrame(
    [{"lipid": n, "category": lipid_cat.get(n, ""), "low": 0.0, "high": 0.5,
      "is_filler": (n == "DSPC")} for n in chosen]
)
comp_edit = st.data_editor(
    comp_df, key="comp_editor", hide_index=True, width="stretch",
    disabled=["lipid", "category"],
    column_config={
        "low": st.column_config.NumberColumn("min fraction", min_value=0.0, max_value=1.0, step=0.01),
        "high": st.column_config.NumberColumn("max fraction", min_value=0.0, max_value=1.0, step=0.01),
        "is_filler": st.column_config.CheckboxColumn("structural / filler"),
    },
)

# ---- process params ----
st.markdown("##### 2. Process parameters & input features")
st.caption("Manufacturing knobs (e.g. flow ratio) **and** measured physicochemical properties used "
           "as model inputs (e.g. liposome size, zeta potential).")
proc_df = pd.DataFrame([
    {"name": "total_lipid_conc", "low": 1.0, "high": 20.0, "unit": "mg/mL"},
    {"name": "flow_ratio", "low": 1.0, "high": 5.0, "unit": "aq:org"},
])
proc_edit = st.data_editor(
    proc_df, key="proc_editor", hide_index=True, width="stretch", num_rows="dynamic",
    column_config={
        "name": st.column_config.TextColumn("parameter"),
        "low": st.column_config.NumberColumn("min"),
        "high": st.column_config.NumberColumn("max"),
        "unit": st.column_config.TextColumn("unit"),
    },
)

# ---- objectives ----
st.markdown("##### 3. Objectives (readouts to optimize)")
obj_default = ["mucus_penetration", "encapsulation_pct", "size_nm"]
obj_df = pd.DataFrame([
    {"readout": r, "direction": READOUT_DEFAULT_DIR.get(r, "max"),
     "target": (100.0 if r == "size_nm" else 0.0), "weight": 1.0}
    for r in obj_default
])
obj_edit = st.data_editor(
    obj_df, key="obj_editor", hide_index=True, width="stretch", num_rows="dynamic",
    column_config={
        "readout": st.column_config.SelectboxColumn("readout", options=READOUT_NAMES),
        "direction": st.column_config.SelectboxColumn("direction", options=["max", "min", "target"]),
        "target": st.column_config.NumberColumn("target (if 'target')"),
        "weight": st.column_config.NumberColumn("weight", min_value=0.0, step=0.5),
    },
)

# ---- constraints ----
st.markdown("##### 4. Constraints (hard requirements)")
con_df = pd.DataFrame([
    {"readout": "pdi", "op": "<=", "bound": 0.3, "bound2": None},
    {"readout": "zeta_mv", "op": "between", "bound": -10.0, "bound2": 10.0},
])
con_edit = st.data_editor(
    con_df, key="con_editor", hide_index=True, width="stretch", num_rows="dynamic",
    column_config={
        "readout": st.column_config.SelectboxColumn("readout", options=READOUT_NAMES),
        "op": st.column_config.SelectboxColumn("operator", options=["<=", ">=", "between"]),
        "bound": st.column_config.NumberColumn("bound"),
        "bound2": st.column_config.NumberColumn("upper (if 'between')"),
    },
)

if st.button("💾 Save campaign", type="primary"):
    errors = []
    comp_rows = comp_edit.to_dict("records")
    if not comp_rows:
        errors.append("Add at least one lipid component.")
    fillers = [r for r in comp_rows if r.get("is_filler")]
    if len(fillers) != 1:
        errors.append("Mark exactly one component as the structural / filler lipid.")
    obj_rows = [r for r in obj_edit.to_dict("records") if r.get("readout")]
    if not obj_rows:
        errors.append("Add at least one objective.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        try:
            cfg = CampaignConfig(
                name=name.strip() or "Untitled campaign",
                description=description.strip(),
                components=[
                    ComponentSpec(lipid=r["lipid"], low=float(r["low"]), high=float(r["high"]),
                                  is_filler=bool(r["is_filler"]))
                    for r in comp_rows
                ],
                process_params=[
                    ProcessParamSpec(name=str(r["name"]), low=float(r["low"]), high=float(r["high"]),
                                     unit=str(r.get("unit") or ""))
                    for r in proc_edit.to_dict("records") if r.get("name")
                ],
                objectives=[
                    ObjectiveSpec(readout=r["readout"], direction=r["direction"],
                                  target=(float(r["target"]) if r.get("target") is not None else None),
                                  weight=float(r.get("weight") or 1.0))
                    for r in obj_rows
                ],
                constraints=[
                    ConstraintSpec(readout=r["readout"], op=r["op"], bound=float(r["bound"]),
                                   bound2=(float(r["bound2"]) if r.get("bound2") is not None else None))
                    for r in con_edit.to_dict("records") if r.get("readout")
                ],
            )
            cid = service.create_campaign(cfg)
            st.session_state["campaign_id"] = cid
            st.success(f"Saved campaign #{cid}: {cfg.name}. Head to **Suggest Experiments**.")
        except Exception as exc:  # validation errors surfaced to the user
            st.error(f"Could not save campaign: {exc}")

# ----------------------------------------------------------------- existing
st.divider()
st.subheader("Existing campaigns")
camps = service.list_campaigns()
if not camps:
    st.info("None yet.")
for c in camps:
    col1, col2, col3 = st.columns([5, 2, 1])
    col1.write(f"**{c['name']}** (#{c['id']})")
    col2.write(f"{c['n_experiments']} experiments")
    if col3.button("Delete", key=f"del_{c['id']}"):
        service.delete_campaign(c["id"])
        st.rerun()
