"""Page 2 — propose the next batch of compositions to run."""
import pandas as pd
import streamlit as st

from mpp import service
from mpp.optimizer import build_optimizer
from mpp.schema import ExperimentRecord
from mpp.ui import apply_theme, page_header, require_campaign

st.set_page_config(page_title="Suggest · MPP", page_icon="🧫", layout="wide")
apply_theme()
camp = require_campaign()
cfg = camp["config"]
page_header("Suggest Experiments", f"Campaign: {cfg.name}", icon="✨")

records = service.records_for_optimizer(camp["id"])
opt = build_optimizer(cfg, rng_seed=42)
n_completed = sum(1 for r in records if opt.is_complete(r))

from mpp.config import MIN_POINTS_FOR_MODEL

if n_completed < MIN_POINTS_FOR_MODEL:
    st.info(
        f"{n_completed} completed experiment(s). Below {MIN_POINTS_FOR_MODEL}, suggestions use a "
        f"**space-filling design** — ideal for seeding your first plate. The model kicks in after that."
    )
else:
    st.success(f"{n_completed} completed experiments — suggestions are now **Bayesian-optimized**.")

n = st.number_input("How many formulations to suggest", min_value=1, max_value=96, value=12, step=1)

if st.button("✨ Suggest next batch", type="primary"):
    with st.spinner("Optimizing…"):
        batch = opt.suggest_batch(records, n=int(n))
    st.session_state["last_batch"] = batch
    st.session_state["last_batch_campaign"] = camp["id"]

batch = st.session_state.get("last_batch") if st.session_state.get("last_batch_campaign") == camp["id"] else None

if batch:
    method = batch[0].get("_method", "?")
    st.markdown(f"**Method:** `{method}`  •  **{len(batch)} suggestions**")

    rows = []
    comp_keys = [c.lipid for c in cfg.components]
    proc_keys = [p.name for p in cfg.process_params]
    wells = [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]
    for i, b in enumerate(batch):
        row = {"well": wells[i] if i < len(wells) else f"X{i}"}
        for k in comp_keys:
            row[f"x:{k}"] = b["composition"].get(k, 0.0)
        for k in proc_keys:
            row[f"p:{k}"] = b["process"].get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Export 96-well worklist (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"worklist_campaign{camp['id']}.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("📋 Save batch as planned experiments"):
            for i, b in enumerate(batch):
                rec = ExperimentRecord(
                    composition=b["composition"], process=b["process"],
                    label=f"suggested ({method})",
                    plate=f"campaign{camp['id']}", well=wells[i] if i < len(wells) else "",
                )
                service.add_experiment(camp["id"], rec, source="suggested", status="suggested")
            st.success(f"Saved {len(batch)} planned experiments. Fill in results on **Upload Results**.")
else:
    st.write("Click **Suggest next batch** to generate compositions.")
