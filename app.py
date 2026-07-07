"""MPP Optimizer — Streamlit entry point / overview."""
import streamlit as st

from mpp import service
from mpp.ui import campaign_selectbox, ensure_init

st.set_page_config(page_title="MPP Optimizer", page_icon="🧫", layout="wide")
ensure_init()

st.title("🧫 MPP Optimizer")
st.caption("ML-driven design of muco-penetrating liposomal nanoparticles — Bayesian optimization over lipid composition.")

camp = campaign_selectbox()

campaigns = service.list_campaigns()
total_exp = sum(c["n_experiments"] for c in campaigns)
completed = 0
if camp:
    exps = service.list_experiments(camp["id"])
    completed = sum(1 for e in exps if e["status"] == "completed")

c1, c2, c3 = st.columns(3)
c1.metric("Campaigns", len(campaigns))
c2.metric("Experiments (all)", total_exp)
c3.metric("Completed (active campaign)", completed)

st.divider()

if camp:
    cfg = camp["config"]
    st.subheader(f"Active campaign: {cfg.name}")
    if cfg.description:
        st.write(cfg.description)
    a, b = st.columns(2)
    with a:
        st.markdown("**Lipid components**")
        for comp in cfg.components:
            tag = " _(structural / filler)_" if comp.is_filler else f"  [{comp.low:.2f}–{comp.high:.2f}]"
            st.markdown(f"- {comp.lipid}{tag}")
        st.markdown("**Process parameters**")
        for p in cfg.process_params:
            st.markdown(f"- {p.name}: {p.low}–{p.high} {p.unit}")
    with b:
        st.markdown("**Objectives**")
        for o in cfg.objectives:
            extra = f" → {o.target}" if o.direction == "target" else ""
            st.markdown(f"- {o.readout}: **{o.direction}**{extra}")
        st.markdown("**Constraints**")
        for con in cfg.constraints:
            rng = f" {con.bound}…{con.bound2}" if con.op == "between" else f" {con.bound}"
            st.markdown(f"- {con.readout} {con.op}{rng}")

st.divider()
st.subheader("How it works")
st.markdown(
    """
1. **Campaign Setup** — choose the lipids in play, their mole-fraction ranges, process
   parameters, and the readouts to optimize (objectives) and respect (constraints).
2. **Suggest Experiments** — the Bayesian optimizer proposes the next batch of compositions.
   With little/no data it returns a space-filling design (seed your first 96-well plate);
   afterwards it proposes informed compositions. Export a 96-well worklist CSV.
3. **Upload Results** — enter the measured readouts for each formulation and attach raw files
   (DLS PDFs, microscopy images, MSD plots, Excel, notes).
4. **Dataset Browser** — view / edit / export the standardized dataset and preview attachments.
5. **Model & Insights** — refit the model to see the Pareto trade-off front, the recommended
   best compositions, sensitivity of the objective to each factor, and predicted-vs-observed fit.

_No data yet?_ Run `python scripts/seed_demo.py` to populate a synthetic demo campaign and watch
the whole loop work end-to-end.
"""
)
