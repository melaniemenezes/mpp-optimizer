"""Page 5 — fit the model and show Pareto front, recommendations, and interpretability."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mpp import service
from mpp.config import MIN_POINTS_FOR_MODEL
from mpp.optimizer import build_optimizer
from mpp.ui import composition_summary, require_campaign

st.set_page_config(page_title="Model & Insights · MPP", page_icon="🧫", layout="wide")
camp = require_campaign()
cfg = camp["config"]
st.title("Model & Insights")
st.caption(f"Campaign: **{cfg.name}**")

records = service.records_for_optimizer(camp["id"])
opt = build_optimizer(cfg, rng_seed=7)
completed = [r for r in records if opt.is_complete(r)]
st.metric("Completed experiments", len(completed))

if not completed:
    st.info("No completed experiments yet. Add results on **Upload Results**.")
    st.stop()

obj_readouts = [o.readout for o in cfg.objectives]

# ----------------------------------------------------------- Pareto front
st.subheader("Pareto trade-off front")
par = opt.pareto(records)
pdf = pd.DataFrame(par["values"])
pdf["pareto"] = ["Pareto-optimal" if p else "dominated" for p in par["is_pareto"]]
pdf["feasible"] = par["feasible"]
pdf["label"] = [r.get("id", "") for r in par["records"]]

if len(obj_readouts) >= 2:
    c1, c2 = st.columns(2)
    xo = c1.selectbox("X objective", obj_readouts, index=0)
    yo = c2.selectbox("Y objective", obj_readouts, index=1)
    fig = px.scatter(
        pdf, x=xo, y=yo, color="pareto", symbol="feasible",
        hover_data=["label"], color_discrete_map={"Pareto-optimal": "#d62728", "dominated": "#9aa0a6"},
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, width="stretch")
else:
    st.bar_chart(pdf.set_index("label")[obj_readouts[0]])
st.caption("Pareto-optimal points are the best achievable trade-offs found so far (feasible = meets all constraints).")

# ----------------------------------------------------- recommended recipes
st.subheader("Recommended best compositions")
best = opt.recommend(records, k=5)
if best:
    rows = []
    for r in best:
        row = {"composition": composition_summary(r["composition"], top=6)}
        for o in cfg.objectives:
            row[o.readout] = r["readouts"].get(o.readout)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.write("—")

# --------------------------------------------------- characterise a readout
st.divider()
st.subheader("Characterise a readout")
st.caption("Pick any measured output (e.g. a diffusion metric) to see which inputs drive it, "
           "how it responds to each, and how well the model predicts it.")
if len(completed) < MIN_POINTS_FOR_MODEL:
    st.info(f"Add at least {MIN_POINTS_FOR_MODEL} completed experiments to unlock this analysis "
            f"(currently {len(completed)}).")
    st.stop()

readout_options = opt.available_readouts(records)
if not readout_options:
    st.info("No numeric readouts recorded yet.")
    st.stop()
target = st.selectbox("Readout to analyse", readout_options, index=0)

with st.spinner("Fitting model…"):
    sens = opt.sensitivities(records, readout=target)
    pvo = opt.predicted_vs_observed(records, readout=target)

st.markdown(f"#### Factor sensitivity for `{sens['readout']}`")
imp = sens["importances"]
if imp:
    idf = pd.DataFrame(sorted(imp.items(), key=lambda kv: kv[1]), columns=["factor", "importance"])
    st.plotly_chart(px.bar(idf, x="importance", y="factor", orientation="h"), width="stretch")
    st.caption("Relative importance from the GP's ARD lengthscales — how strongly each input moves this readout.")

st.markdown("#### Partial dependence (top factors)")
pds = sens["partial_dependence"]
if pds:
    cols = st.columns(min(2, len(pds)))
    for i, (dim, curve) in enumerate(pds.items()):
        fig = px.line(x=curve["x"], y=curve["y"], labels={"x": dim, "y": sens["readout"]})
        cols[i % len(cols)].plotly_chart(fig, width="stretch")
    st.caption("Predicted readout as each input varies with the others held at their mean.")

st.subheader(f"Predicted vs observed — `{pvo['readout']}` (leave-one-out)")
if pvo["obs"]:
    vdf = pd.DataFrame({"observed": pvo["obs"], "predicted": pvo["pred"]})
    lo = min(vdf.min().min(), 0)
    hi = vdf.max().max()
    fig = px.scatter(vdf, x="observed", y="predicted")
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal",
                             line=dict(dash="dash", color="gray")))
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("How well the surrogate predicts held-out experiments. Points near the dashed line = good fit.")
