"""Page 4 — browse, export, and inspect the standardized dataset."""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from mpp import service, storage
from mpp.ui import apply_theme, composition_summary, page_header, require_campaign

st.set_page_config(page_title="Dataset · MPP", page_icon="🧫", layout="wide")
apply_theme()
camp = require_campaign()
page_header("Dataset Browser", f"Campaign: {camp['config'].name}", icon="🗂️")

df = service.experiments_dataframe(camp["id"])
if df.empty:
    st.info("No experiments yet. Add results on **Upload Results**.")
    st.stop()

status_filter = st.multiselect("Status", sorted(df["status"].unique()), default=list(df["status"].unique()))
view = df[df["status"].isin(status_filter)] if status_filter else df

st.dataframe(view, hide_index=True, width="stretch")
st.caption(f"{len(view)} of {len(df)} experiments shown.")

col1, col2 = st.columns(2)
with col1:
    st.download_button("⬇️ Export CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name=f"mpp_dataset_campaign{camp['id']}.csv", mime="text/csv")
with col2:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        view.to_excel(writer, index=False, sheet_name="experiments")
    st.download_button("⬇️ Export Excel", buf.getvalue(),
                       file_name=f"mpp_dataset_campaign{camp['id']}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.subheader("Inspect an experiment")
exp_id = st.selectbox("Experiment", view["id"].tolist())
exp = service.get_experiment(int(exp_id))
if exp:
    a, b = st.columns(2)
    with a:
        st.markdown(f"**#{exp['id']}** · {exp['label'] or '—'}  \nStatus: `{exp['status']}` · Source: `{exp['source']}`")
        st.markdown(f"Plate/Well: {exp['plate'] or '—'} / {exp['well'] or '—'}")
        st.markdown("**Composition:** " + (composition_summary(exp["composition"], top=8) or "—"))
        st.markdown("**Process:** " + (", ".join(f"{k}={v}" for k, v in exp["process"].items()) or "—"))
        st.markdown("**Readouts:** " + (", ".join(f"{k}={v}" for k, v in exp["readouts"].items()) or "—"))
        if exp["notes"]:
            st.markdown(f"**Notes:** {exp['notes']}")
    with b:
        st.markdown(f"**Attachments ({len(exp['attachments'])})**")
        for att in exp["attachments"]:
            kind = storage.kind(att["filename"])
            with st.expander(f"{att['filename']}  ·  {att['size']} bytes  ·  {kind}"):
                p = att["path"]
                exists = Path(p).exists()
                if not exists:
                    st.warning("File missing on disk.")
                elif kind == "image":
                    st.image(p)
                elif kind == "text":
                    try:
                        st.code(Path(p).read_text(errors="replace")[:3000])
                    except Exception:
                        st.write("(could not read text)")
                elif kind == "pdf":
                    preview = storage.pdf_text_preview(p)
                    st.write(preview or "(no extractable text — download to view)")
                elif kind == "excel":
                    try:
                        st.dataframe(pd.read_excel(p).head(50), width="stretch")
                    except Exception as e:
                        st.write(f"(could not preview: {e})")
                if exists:
                    st.download_button("Download", Path(p).read_bytes(), file_name=att["filename"],
                                       key=f"dl_{att['id']}")

    if st.button("🗑️ Delete this experiment", key=f"delexp_{exp['id']}"):
        service.delete_experiment(int(exp_id))
        st.rerun()
