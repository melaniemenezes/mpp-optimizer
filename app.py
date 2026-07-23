"""MPP Optimizer — Streamlit entry point / overview."""
import streamlit as st

from mpp import service
from mpp.ui import apply_theme, campaign_selectbox, ensure_init, page_header

st.set_page_config(page_title="MPP Optimizer", page_icon="🧫", layout="wide")
apply_theme()
ensure_init()

page_header(
    "MPP Optimizer",
    "ML-driven design of muco-penetrating liposomal nanoparticles — Bayesian optimization over lipid composition.",
    icon="🧫",
)

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
c3.metric("Completed · active campaign", completed)

st.write("")

if camp:
    cfg = camp["config"]
    with st.container(border=True):
        st.subheader(cfg.name)
        if cfg.description:
            st.caption(cfg.description)
        a, b = st.columns(2)
        with a:
            st.markdown("**Lipid components**")
            for comp in cfg.components:
                tag = " · _structural / filler_" if comp.is_filler else f"  `[{comp.low:.2f}–{comp.high:.2f}]`"
                st.markdown(f"- {comp.lipid}{tag}")
            if cfg.process_params:
                st.markdown("**Process parameters & input features**")
                for p in cfg.process_params:
                    st.markdown(f"- {p.name}: {p.low}–{p.high} {p.unit}")
        with b:
            st.markdown("**Objectives**")
            for o in cfg.objectives:
                extra = f" → {o.target}" if o.direction == "target" else ""
                st.markdown(f"- {o.readout}: **{o.direction}**{extra}")
            if cfg.constraints:
                st.markdown("**Constraints**")
                for con in cfg.constraints:
                    rng = f" {con.bound}…{con.bound2}" if con.op == "between" else f" {con.bound}"
                    st.markdown(f"- {con.readout} {con.op}{rng}")
else:
    st.info("No active campaign yet — create one on **Campaign Setup**, or load a ready-made one there.")

st.write("")

with st.container(border=True):
    st.subheader("How it works")
    st.markdown(
        """
1. **Campaign Setup** — choose the lipids, their mole-fraction ranges, input features, and the
   readouts to optimize (objectives) and respect (constraints).
2. **Suggest Experiments** — the optimizer proposes the next batch of compositions (space-filling
   at cold start, Bayesian afterwards). Export a 96-well worklist CSV.
3. **Upload Results** — enter measured readouts and attach raw files (Excel, PDF, images, notes).
4. **Dataset Browser** — view / export the standardized dataset and preview attachments.
5. **Model & Insights** — Pareto front, recommended compositions, and per-readout characterisation
   (which inputs drive each output, partial dependence, predicted-vs-observed).
        """
    )
    st.caption("New here? Generate the full project guide PDF with `python scripts/make_guide_pdf.py` "
               "— the science, the algorithm explained from scratch, and a walkthrough.")
